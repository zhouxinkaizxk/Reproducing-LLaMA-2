import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from typing import Any,Optional,Tuple
from dataclasses import dataclass
import math

from transformers import PreTrainedModel, AutoTokenizer
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast



class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768,#模型主干的向量维度
            n_layers: int = 12,
            n_heads: int = 16,
            n_kv_heads: int = 8,
            vocab_size: int = 6144,
            hidden_dim: int = None,#MLP内部的中间维度
            multiple_of: int = 64,#自动计算MLP中间维度时，向上取整为64的倍数，便于计算效率优化
            norm_eps: float = 1e-5,
            max_seq_len: int = 512,
            dropout: float = 0.0,
            flash_attn: bool = True,
            pad_token_id: int = 0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.pad_token_id = pad_token_id

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-8):
        super().__init__()
        #eps是为了防止除以0的情况
        self.eps = eps
        #weight是一个可训练的参数，初始化为全1，维度为dim
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self,x):
        #计算RMSNorm的核心部分
        #x.pow(2).mean(-1, keepdim=True)计算了输入x的平方的均值
        #torch.rsqrt是平方根的倒数，这样就得到了RMSNorm的分母部分，再加上eps防止分母为0
        #最后乘以x，得到RMSNorm的结果
        #沿着最后一维计算均值，保持维度不变，并利用广播机制，对每个token的向量进行表准化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        #forward函数是模型的前向传播函数
        #首先将输入x转换成float类型，然后进行RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight，这是RMSNorm的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

#获得旋转嵌入的实部和虚部
#注意：此处的维度应为dim//n_heads，因为我们是对每个head进行旋转嵌入
def precompute_freqs_cis(dim: int, end: int,theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    #这步是在计算token维度的频率
    freqs = 1.0 / (theta ** (torch.arange(0,dim,2)[:(dim // 2)].float() / dim))
    #生成一个从0到end的序列，长度为end
    #这步是在计算token在序列中的位置
    t = torch.arange(end, device=freqs.device)
    #计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    #得到实部和虚部，分别是cos(freqs)和sin(freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

#此函数的作用是将freqs_cis调整为与输入x的形状相匹配，以便在后续的计算中进行广播操作
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    #构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    #将查询和键张量转换为float类型，并重塑形状以分离实部和虚部
    #reshape(xq.shape[:-1] + (-1, 2))维度扩展
    #unbind(-1)将最后一维分离为两个张量，分别是实部和虚部                           
    xq_r,xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r,xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    #应用旋转嵌入公式
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    #将最后两个维度合并，并还原为原始张量的形状
    #torch.stack([xq_out_r, xq_out_i],dim=-1)在最后新增一个维度，将实部和虚部配成一对
    #flatten(3)从索引为3的维度开始，把后面的维度和并
    xq_out = torch.stack([xq_out_r, xq_out_i],dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i],dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x 是 Key 张量 xk，或者 Value 张量 xv
    bs,slen,n_kv_heads,head_dim = x.shape

    # 如果重复次数为1，则不需要重复，直接返回原始张量
    if n_rep == 1:
        return x

    #对张量进行扩展和重塑操作以重复键值对
    return(
        x[:,:,:,None,:] # 在第四个维度添加一个新的维度
        .expand(bs,slen,n_kv_heads,n_rep,head_dim) # 将新添加的维度扩展到n_rep大小，实现重复的效果
        .reshape(bs,slen,n_kv_heads*n_rep,head_dim) # 将重复后的张量重塑为(bs, slen, n_kv_heads*n_rep, head_dim)的形状
    )

class Attention(nn.Module):
    def __init__(self,args: ModelConfig):
        super().__init__()
        #根据是否指定n_kv_heads用于确定key value头的数量
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        #模型并行处理大小，默认为1
        model_parallel_size = 1
        #本地计算头数，等于总头数除以模型并行处理大小
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        #重复次数，用于扩展键和值的尺寸
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        #每个头的维度，等于模型维度除以头的总数
        self.head_dim = args.dim // args.n_heads

        #定义权重矩阵
        self.wq = nn.Linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.wk = nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        self.wv = nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        #输出权重矩阵,融合所有注意力头的输出
        self.wo = nn.Linear(args.n_heads*self.head_dim,args.dim,bias=False)

        #定义dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        #检查是否使用Flash Attention
        self.flash =self.flash = args.flash_attn and hasattr(torch.nn.functional,'scaled_dot_product_attention') 

        if not self.flash:
            # 若不支持Flash Attention，则使用手动实现的注意力机制，并设置mask。
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建一个上三角矩阵，用于遮蔽未来信息。
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self,x: torch.Tensor,freqs_cos: torch.Tensor,freqs_sin: torch.Tensor,attention_mask: Optional[torch.Tensor]=None):
        #获取批次大小和序列长度,[batch_size, seq_len, dim]
        bsz,seqlen,_ = x.shape

        #计算查询、键和值张量
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x) 

        # 调整形状以适应头的维度。
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        #应用旋转嵌入(RoPE)
        xq,xk = apply_rotary_emb(xq,xk,freqs_cos,freqs_sin)

        #对键和值进行扩展以适应重复次数
        xk = repeat_kv(xk,self.n_rep)
        xv = repeat_kv(xv,self.n_rep)

        # 将头作为批次维度处理。
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)

        #根据是否支持Flash Attention，选择不同的实现方式
        if self.flash:
            # 使用Flash Attention
            if key_padding_mask is not None:
                causal_mask = torch.ones((seqlen,seqlen),dtype=torch.bool,device=x.device).tril()
                full_attn_mask = causal_mask & key_padding_mask
                output = torch.nn.functional.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=full_attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
            else:
                output = torch.nn.functional.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )  
        else:
            # 使用手动实现的注意力机制
            scores = torch.matmul(xq,xk.transpose(2,3))/math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:,:,:seqlen,:seqlen]
            if key_padding_mask is not None:
                #~表示取反
                #.masked_fill()，将~key_padding_mask中为True的元素替换为float('-inf')
                #即把mask位置元素设置为float('-inf')
                scores = scores.masked_fill(~key_padding_mask, float('-inf'))
            scores = F.softmax(scores.float(),dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        #恢复时间维度并合并头
        #[bsz,n_local_heads,seqlen,head_dim) -> [bsz,seqlen,n_heads,head_dim) -> [bsz,seqlen,n_heads*head_dim)]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        #最后投影回残差流
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class MLP(nn.Module):
    def __init__(self,dim: int,hidden_dim: int,multiple_of: int,dropout: float):
        super().__init__()
        #如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        #然后将其减少到2/3，最后确保它是mutiple_of的倍数
        if hidden_dim is None:
            hidden_dim = dim * 4
            hidden_dim = int(2*hidden_dim/3)
            hidden_dim = multiple_of * ((hidden_dim + mutiple_of - 1)//mutiple_of)
        #定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三层线性变换，从输入维度到隐藏维度
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和SILU激活函数
        # 然后，结果乘以输入x通过第三层线性变换的结果
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        # 定义多头注意力的头数
        self.n_heads = args.n_heads
        # 定义输入维度
        self.dim = args.dim
        # 定义每个头的维度，等于输入维度除以头数
        self.head_dim = args.dim // args.n_heads
        # 定义LLaMA2Attention对象，用于进行多头注意力计算
        self.attention = Attention(args)
        # 定义LLaMAMLP对象，用于进行前馈神经网络计算
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )       
        # 定义层的ID
        self.layer_id = layer_id
        # 定义注意力计算的归一化层
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义前馈神经网络计算的归一化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, attention_mask: Optional[torch.Tensor] = None):
        # 前向传播函数
        # 首先，输入x经过注意力归一化层，然后进行注意力计算，结果与输入x相加得到h
        # 然后，h经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与h相加得到输出
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin, attention_mask=attention_mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(PreTrainedModel):
    config_class = ModelConfig  # 配置类
    last_loss: Optional[torch.Tensor] # 记录最后一次计算的损失
    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        # 初始化模型参数
        self.args = args
        # 词汇表大小
        self.vocab_size = args.vocab_size   
        # 层数
        self.n_layers = args.n_layers
        # 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout层
        self.dropout = nn.Dropout(args.dropout)
        # Decoder层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        # 归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 将词嵌入层权重和输出层权重共享
        self.tok_embeddings.weight = self.output.weight

        # 预计算相对位置嵌入的频率
        freqs_cos,freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos",freqs_cos,persistent=False)
        self.register_buffer("freqs_sin",freqs_sin,persistent=False)

        # 初始化所有权重
        # self.apply()会递归遍历模型及其内部的子模块，把每个模块交给哦_init_weights处理一次
        self.apply(self._init_weights)

        # 对残差投影进行特殊的缩放初始化
        # self.named_parameters()每次返回一对参数名字和参数对象
        # endwith()是python的字符串方法，意思是”是否以...结尾”
        for pn,p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p,mean=0,std=0.02/math.sqrt(2*args.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # 输出容器
        self._no_split_modules = [name for name, _ in self.named_modules()]  # 不分割的模块列表

    def _init_weights(self, module):
        # 初始化权重的函数
        # isinstance(对象, 类型),判断对象是否属于这个类型
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _prepare_attention_mask(self, attention_mask: Optional[torch.Tensor], tokens: torch.Tensor) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        # 如果attention_mask是4维或者3维，将其展平为2维
        if attention_mask.dim() == 4:
            attention_mask = attention_mask[:, 0, 0, :]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask[:, 0, :]
        attention_mask = attention_mask.to(tokens.device)
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask > 0
        if attention_mask.shape != tokens.shape:
            raise ValueError(f"attention_mask shape {attention_mask.shape} must match input_ids shape {tokens.shape}")
        return attention_mask


    # 用来整理idx和掩码，根据attention_mask取出每条序列的有效token，然后把较短的序列在左边补齐
    def _left_pad_by_attention_mask(
            self,
            idx: torch.Tensor, # 一批输入文本分词后对应的token ID
            attention_mask: Optional[torch.Tensor],
            pad_token_id: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # .all()是PyTorch张量的方法，检查张量中所有元素是否都为True，只要有1个元素为False，就返回False
        # 反之，如果所有元素都为True，就返回True
        if attention_mask is None or attention_mask.all():
            return idx, attention_mask

        bsz = idx.size(0)
        lengths = attention_mask.long().sum(dim=1)
        max_len = max(int(lengths.max().item()), 1)
        # .new_full()创建一个新张量:已有张量.new_full(新张量的形状, 填充值)，数据类型与设备与原有张量相同
        packed_idx = idx.new_full((bsz, max_len), pad_token_id)
        packed_mask = attention_mask.new_zeros((bsz, max_len), dtype=torch.bool)

        for row in range(bsz):
            valid_len = int(lengths[row].item())
            if valid_len <= 0:
                continue
            valid_tokens = idx[row][attention_mask[row]]
            packed_idx[row, max_len - valid_len:] = valid_tokens
            packed_mask[row, max_len - valid_len:] = True

        return packed_idx, packed_mask

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        - tokens: Optional[torch.Tensor], 输入 token 张量。
        - targets: Optional[torch.Tensor], 目标 token 张量。
        - kv_cache: bool, 是否使用键值缓存。
        - kwargs: 其他关键字参数。

        - self.OUT: CausalLMOutputWithPast, 包含 logits 和损失。
        """

        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'labels' in kwargs:
            targets = kwargs['labels']
        attention_mask = self._prepare_attention_mask(kwargs.get('attention_mask'), tokens)

        # 前向传播函数
        _bsz, seqlen = tokens.shape
        # 通过词嵌入层和Dropout层
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        # 获取相对位置嵌入的频率
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 通过Decoder层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, attention_mask=attention_mask)
        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            # 如果给定了目标，计算损失
            logits = self.output(h)
            ignore_index = self.args.pad_token_id if self.args.pad_token_id is not None else 0
            # torch.any(targets == -100)判断张量targets里是否含有 -100，若有怎返回True，否则返回False
            if torch.any(targets == -100):
                ignore_index = -100
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
                reduction='none'
            )
        else:
            # 推理时的小优化：只对最后一个位置的输出进行前向传播
            # attention_mask管输入里哪些位置是PAD，把句子补成等长，attention_mask记录补了哪些位置
            if attention_mask is None:
                logits = self.output(h[:, [-1], :])
            else:
                full_logits = self.output(h)    
                # .clamp(min=1)作用是将张量中小于1的元素全部变成1，大于等于1的元素则不变
                last_token_pos = attention_mask.long().clamp(min=1) - 1
                # 用到了PyTorch的高级索引来取每个bsz中最后一个有效token的输出，其输出即是预测的下一个token
                logits = full_logits[torch.arange(_bsz,device=tokens.device),last_token_pos].unsqueeze(1)
            self.last_loss = None

        # 设置输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(
            self,
            idx,
            stop_id=None,
            max_new_tokens=256,
            temperature=1.0,
            top_k=None,
            attention_mask: Optional[torch.Tensor] = None,
            pad_token_id: Optional[int] = None
    ):
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
        """
        if pad_token_id is None:
            pad_token_id = self.args.pad_token_id if self.args.pad_token_id is not None else 0
        attention_mask = self._prepare_attention_mask(attention_mask, idx)
        idx, attention_mask = self._left_pad_by_attention_mask(idx, attention_mask, pad_token_id)

        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            mask_cond = None
            if attention_mask is not None:
                mask_cond = attention_mask if attention_mask.size(1) <= self.args.max_seq_len else attention_mask[:, -self.args.max_seq_len:]
            
            # 前向传播获取序列中最后一个位置的 logits
            logits = self(idx_cond, attention_mask=mask_cond).logits
            logits = logits[:, -1, :] # 只保留最后一个时间步的输出
            
            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                # torch.multinomial()：根据输入的概率分布 probs，进行随机采样，抽取一个元素的索引，作为下一个 Token ID。
                idx_next = torch.multinomial(probs, num_samples=1)

            prev_finished = finished.clone()
            if stop_id is not None:
                # .any():检查张量prev_finished中是否有任何元素为True，若有则返回True，否则返回False。
                if prev_finished.any():
                    fill_token = pad_token_id if pad_token_id is not None else stop_id
                    # torch.where(condition, A, B)，condition为True选A，为False选B
                    idx_next = torch.where(prev_finished[:, None], torch.full_like(idx_next, fill_token), idx_next)
                # .eq():逐元素比较，相等的元素返回True，否则返回False。
                # "|"表示逐元素逻辑或
                finished = prev_finished | idx_next[:, 0].eq(stop_id)

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)
            if attention_mask is not None:
                next_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
                if prev_finished.any():
                    next_mask[prev_finished] = False
                attention_mask = torch.cat((attention_mask, next_mask), dim=1)

            if stop_id is not None and finished.all():
                break

        return idx[:, index:] # 只返回生成的token

        
    def _random_sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        随机采样：基于概率分布随机选择token

        Args:
            logits: 模型输出的logits，形状为 (batch_size, vocab_size)
            temperature: 温度参数，控制随机性
            top_k: 只考虑概率最高的k个token

        Returns:
            选择的token索引，形状为 (batch_size, 1)
        """
        # 缩放 logits
        logits = logits / temperature

        # 应用top-k过滤
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # 将不在 top-k 内的 logits 设为负无穷
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 计算概率并采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        return idx_next

    def _beam_search(self, idx: torch.Tensor, max_new_tokens: int, num_beams: int,
                     temperature: float = 1.0, top_k: int = None, stop_id: int = None) -> torch.Tensor:
        """
        束搜索：维护多个候选序列，选择最优路径

        束搜索的核心思想：在每一步生成时，不是只选择一个最佳token，
        而是保留多个候选路径，最终选择累积概率最高的完整序列。

        Args:
            idx: 输入序列，形状为 (batch_size, seq_len)
            max_new_tokens: 最大生成token数量
            num_beams: 束宽度，表示保留的候选路径数量
            temperature: 温度参数，控制分布的平滑程度
            top_k: top-k过滤参数，限制候选token范围
            stop_id: 停止生成的token ID，遇到则停止

        Returns:
            生成的token序列，形状为 (batch_size, generated_length)
            只返回新生成的部分，不包含原始输入序列
        """
        # 获取输入序列的基本信息
        batch_size = idx.shape[0]  # 批次大小，通常为1
        seq_len = idx.shape[1]     # 输入序列长度

        # 初始化束：创建 num_beams 个候选序列
        beams = [idx.clone() for _ in range(num_beams)]
        # 初始化每个候选序列的累积对数概率分数
        beam_scores = torch.zeros(num_beams, device=idx.device)
        # 第一个候选是原始输入序列，分数为0
        beam_scores[0] = 0.0
        # 其他候选初始分数设为负无穷，表示尚未生成
        beam_scores[1:] = float('-inf')

        # 主循环：逐步生成新的token，最多生成 max_new_tokens 个
        for step in range(max_new_tokens):
            # 每轮迭代收集新的候选序列和分数
            new_beams = []   # 新的候选序列列表
            new_scores = []  # 对应的分数列表

            # 遍历当前的所有候选序列
            for beam_idx, beam in enumerate(beams):
                # 跳过无效候选（分数为负无穷的序列）
                if beam_scores[beam_idx] == float('-inf'):
                    continue

                # 序列长度检查：如果超过最大长度，截取最后的部分
                beam_cond = beam if beam.size(1) <= self.args.max_seq_len else beam[:, -self.args.max_seq_len:]

                # 前向传播：获取模型对当前序列的预测
                output = self(beam_cond)
                # 提取最后一个位置的logits，用于预测下一个token
                logits = output.logits[:, -1, :]  # 形状: (1, vocab_size)

                # 温度缩放：调整logits的分布
                if temperature != 1.0:
                    logits = logits / temperature
                    # 温度 > 1：分布更平滑，增加随机性
                    # 温度 < 1：分布更尖锐，更确定

                # Top-k过滤：限制候选token的范围，提高质量
                if top_k is not None:
                    # 找到logits中前top_k个最大的值
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # 将不在前top_k内的logits设为负无穷
                    logits[logits < v[:, [-1]]] = -float('Inf')
                    # 这样采样时只会考虑前top_k个token

                # 计算对数概率：使用log_softmax避免数值不稳定
                # F.log_softmax()先对输入求softmax，得到概率分布，再对概率取自然对数
                log_probs = F.log_softmax(logits, dim=-1)

                # 获取前 num_beams 个最可能的候选token
                # 注意：这里的top-k与上面的top-k不同
                # 上面的top-k是全局过滤，这里是束搜索的分支选择
                top_log_probs, top_indices = torch.topk(log_probs, k=num_beams, dim=-1)

                # 为当前候选序列生成 num_beams 个扩展序列
                for k in range(num_beams):
                    # 选择第k个候选token
                    token = top_indices[:, k:k+1]      # token ID
                    log_prob = top_log_probs[:, k]     # 对应的对数概率

                    # 扩展序列：将新token添加到当前序列末尾
                    new_beam = torch.cat([beam, token], dim=1)
                    # 更新累积分数：原序列分数 + 新token的对数概率
                    new_score = beam_scores[beam_idx] + log_prob.item()

                    # 保存新的候选序列和分数
                    new_beams.append(new_beam)
                    new_scores.append(new_score)

            # 安全检查：如果没有生成任何有效候选，提前结束
            if not new_beams:
                break

            # 筛选最佳候选：从所有新生成的候选中选择分数最高的 num_beams 个
            # 按分数降序排序，获取索引
            sorted_indices = sorted(range(len(new_scores)), key=lambda i: new_scores[i], reverse=True)
            # 选择前 num_beams 个最佳候选
            beams = [new_beams[i] for i in sorted_indices[:num_beams]]
            beam_scores = [new_scores[i] for i in sorted_indices[:num_beams]]

            # 停止条件检查：检查最佳序列是否以停止token结尾
            if stop_id is not None and beams[0][0, -1] == stop_id:
                break

        # 返回得分最高的序列，只返回新生成的部分（去掉原始输入）
        # beams[0] 是最终得分最高的完整序列
        # [:, seq_len:] 切片只保留生成部分
        return beams[0][:, seq_len:]
