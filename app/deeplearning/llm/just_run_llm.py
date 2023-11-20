import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/youri-7b")
model = AutoModelForCausalLM.from_pretrained("rinna/youri-7b")

if torch.cuda.is_available():
    model = model.to("cuda")

text = "最近のAI動向は、"
token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=200,
        min_new_tokens=200,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)
"""
西田幾多郎は、プラトンの復権を主張し、対する従来の西洋哲学は、近代の合理主義哲学に委ね、「従来の哲学は破 壊されてしまった」と述べている。 西田幾多郎は、西洋近代哲学の「徹底的な検討」を拒んだ。それは、「現代的理解の脆弱性を補う筈の、従来のヨーロッパに伝わる哲学的な方法では到底それができなかったからである」とい
"""
