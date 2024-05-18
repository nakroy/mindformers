import mindspore as ms
from mindformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor

# 指定图模式，指定使用训练卡id
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# 以下两种tokenizer实例化方式选其一即可
# 1. 在线加载方式
tokenizer = AutoTokenizer.from_pretrained("glm3_6b")
# 2. 本地加载方式
# tokenizer = AutoProcessor.from_pretrained("/path/to/your.yaml").tokenizer

# 以下两种model的实例化方式选其一即可
# 1. 直接根据默认配置实例化
# model = AutoModel.from_pretrained('glm3_6b')
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained('glm3_6b')
config.use_past = True                  # 此处修改默认配置，开启增量推理能够加速推理性能
config.seq_length = 2048                      # 根据需求自定义修改其余模型配置
config.checkpoint_name_or_path = "/path/to/your.ckpt"
model = AutoModel.from_config(config)   # 从自定义配置项中实例化模型

role="user"

inputs_list=["你好", "请介绍一下华为"]

for input_item in inputs_list:
    history=[]
    inputs = tokenizer.build_chat_input(input_item, history=history, role=role)
    inputs = inputs['input_ids']
    # 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
    outputs = model.generate(inputs, do_sample=False, top_k=1, max_length=config.seq_length)
    response = tokenizer.decode(outputs)
    for i, output in enumerate(outputs):
        output = output[len(inputs[i]):]
        response = tokenizer.decode(output)
        print(response)
    # answer 1:
    # 你好👋！我是人工智能助手 ChatGLM3-6B，很高兴见到你，欢迎问我任何问题。

    # answer 2:
    # 华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。该公司也在智能手机、电脑、平板电脑、云计算等领域开展业务,其产品和服务覆盖全球170多个国家和地区。

    # 华为的主要业务包括电信网络设备、智能手机、电脑和消费电子产品。公司在全球范围内有超过190,000名员工,其中约一半以上从事研发工作。华为以其高品质的产品和服务赢得了全球客户的信任和好评,也曾因其领先技术和创新精神而获得多项国际奖项和认可。

    # 然而,华为也面临着来自一些国家政府的安全问题和政治压力,其中包括美国政府对其产品的禁令和限制。华为一直坚称自己的产品是安全的,并采取了一系列措施来确保其产品的安全性和透明度。