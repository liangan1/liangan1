## Welcom to Liangang's Github

Liangang is an AI framework engineer in Intel and now is working on the LLM inference optimization. 

## My Contributon in Github

[Tensor Parallel for LLM](https://github.com/intel/intel-extension-for-pytorch/commit/4fa64459d03a17839ec49d1081e9c7e15e0c7f52) 

Implementate the tensor parallel from the scratch and use Shared Memory based All-reduce to speedup.  

[PagedAttention](https://github.com/intel/intel-extension-for-pytorch/commits/main/csrc/cpu/aten/kernels/PagedAttentionKrnl.cpp)

This kernel enables the flash decoding based the paged kv cache and it has been used in the [vllm](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/ipex_attn.py) repository. 

[Indirect Access KV Cache](https://github.com/intel/intel-extension-for-pytorch/blob/main/csrc/cpu/aten/kernels/MaskedMultiHeadAttentionKrnl.cpp) 

KV cache is used to reduce computation for decoder layer but it also brings memory overheads. For example, when we use beam search, the kv_cache should be reordered according to latest beam idx and the current key/value should also be concat with kv_cache in the attention layer to get entire context to do scale dot product. When the sequence is very long, memory overheads caused by the reorder_cache and concat will be performance bottleneck. Indirect Access KV_cache (IAKV) is provided to reduce these overheads. Firstly, IAKV pre-allocates buffers (key and value use different buffer) to store all key/value hidden states and beam index information, the data format is shown in the following left figure (beam_width=4 in this case) and token state of key (value) in every timestamp will be store in this pre-allocated buffer. Secondly, we can use beam index history which is shown in the following right figure to decide which beam should be used by a timestamp and this information will generate a offset to access the kv_cache buffer which means that the reorder_cache and concat overheads will be eliminated by this way.

[Rotary Position Embeeding](https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/blob/cpu-device/csrc/cpu/aten/kernels/RotaryPositionEmbeddingKnl.cpp) 

Support multple LLM models. e.g., lamma/gpt-neox/falcon/GPT-J 6B/CodeGen/ChatGLM...

More contiribution can be found [here](https://github.com/intel/intel-extension-for-pytorch/graphs/contributors)

## My Publications and Talks 

[基于至强处理器的AI软件生态](https://marketing.csdn.net/p/4f3a7da76a0dc06a0db8a1f251dd9eea?pId=2409)

[A Novel Scale-Out Training Solution for Deep Learning Recommender Systems](https://www.intel.com/content/www/us/en/developer/articles/technical/novel-scale-out-training-solution-deep-learning-recommender-systems.html)

