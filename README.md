# api2.py 适用于 GPT-SoVITS 的api调用接口

>  [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/README.md) 一个非常棒的零(少)样本中文声音克隆项目，之前有一篇文章详细介绍过如何部署和训练自己的模型([点击查看](https://juejin.cn/post/7341210909070000168))，并使用该模型在web界面中合成声音，可惜它自带的 api.py 在调用方面支持比较差，比如不能中英混合、无法按标点切分句子等，因此对api.py做了修改，详细使用说明如下。
> 
> 修改后代码开源地址：https://github.com/jianchang512/gptsovits-api




下载api2.py，复制到GPT-SoVITS软件目录下，执行命令同自带api.py一样，只需要将名字 api.py 改成 api2.py。默认端口也是 9880，默认绑定 127.0.0.1

----


## 使用默认模型启动并指定默认参考音频 -dr -dt -dl

假设参考音频要使用根目录下的 **123.wav** ,音频文字是 **“一二三四五六七。”** ，音频语言是中文，那么命令如下：


` .\runtime\python api2.py -dr "123.wav" -dt "一二三四五六七。" -dl "zh" `


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a131add45289448391eeec4ebcd9c2d1~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1018&h=82&s=9175&e=png&b=0c0c0c)

Linux下命令去掉 `.\runtime\` 即可

如上述命令这样在启动后，指定的参考音频将作为默认配置，当请求api数据中如果未指定参考音频，将使用。


## 默认模型启动并绑定ip地址和端口 -a -p

假设要指定绑定内网ip **192.168.0.120** ，端口要使用 **9001**，不指定默认参考音频，那么执行如下命令:

`.\runtime\python api2.py -a "127.0.0.1" -p 9001 `


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4d57b23134b5436f825f04944e4bc250~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=700&h=70&s=4954&e=png&b=0c0c0c)


## 启动自己训练好的模型 -s -g

在指定自己的模型时，必须确保同时**指定参考音频**。训练好的模型可分别在软件目录下的 **GPT_weights** 和 **SoVITS_weights** 目录下寻找，以你训练时命名的模型名称开头，后跟`e数字`最大的那个即可。

`.\runtime\python api2.py -s "SoVITS_weights/你的模型名" -g "GPT_weights/你的模型名" -dr "参考音频路径和名称" -dt "参考音频的文字内容，使用双引号括起来，确保文字内容里没有双引号" -dl zh|ja|en三者选一 `


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/81051eca33694c50a40ddfd67828eda4~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1666&h=54&s=10806&e=png&b=0c0c0c)


## 强制在CPU上推理 -d cpu

默认将优先使用 CUDA 或 mps(Mac), 如果你想指定在CPU上运行，可以通过 `-d cpu `指定

`.\runtime\python api2.py -d cpu` 

注意 `-d` 后只能是 `cpu 或 cuda 或 mps`，并且只有在正确配置 cuda 后才能指定 cuda，只有Apple CPU Mac上才能指定 mps


## 全部按照默认运行

`.\runtime\python api2.py`

这种方式将使用默认模型，并且在 api 请求时必须指定参考音频、参考音频文字内容、参考音频语言代码，api 监听 9880 端口


## 可使用的语言代码 -dl

仅支持 **中文、日语、英语** 三种语言，对应只可使用 `zh(代表中文或中英混合)`、`ja(代表日语或日英混合)`、`en(代表英语)`，使用 -dl 指定，如 `-dl zh`,`-dl ja`,`-dl en`

## 参考音频路径 -dr

参考音频填写以软件根目录为起点的相对目录，假如你的参考音频是直接放在软件根目录下，那么只需要填写带后缀的完整名字即可，比如 `-dr 123.wav`,如果是在子目录下，比如在 `wavs` 文件夹下，那么填写 `-dr "wavs/123.wav"`

## 参考音频的文字内容 -dt

参考音频的文字内容就是音频里的说话文字，需要正确填写标点符号，并使用英文双引号括起来。请注意，文字中不要再有英文双引号。

`-dt "这里填写参考音频的文字内容，不要含有英文双引号"`


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/075dcfcb088f45b0982420638f1005b0~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1448&h=750&s=148467&e=png&b=0c0c0c)


## 可用的命令行参数:

**模型相关参数**

`-s`  SoVITS模型路径, 默认模型无需填写，自训练模型在 SoVITS_weights 目录下

`-g`  GPT模型路径, 默认模型无需填写，自训练模型在 GPT_weights 目录下

**参考音频相关参数**

`-dr`  默认参考音频路径，如果在根目录下，直接填写带后缀名字，否则加上 路径/名字

`-dt`  默认参考音频文本，音频的文字内容，以英文双引号括起来

`-dl`  默认参考音频内容的语种, "zh"或"en"或"ja"

**设备和地址相关参数**

`-d`  推理设备, "cuda","cpu","mps"  只有配置好了cuda环境才可指定cuda，只有Apple CPU上才可指定mps

`-a`  绑定地址, 默认"127.0.0.1"

`-p`  绑定端口, 默认9880

**不常用参数，新手可忽略不必设置**

`-fp`  使用全精度

`-hp`  使用半精度

`-hb`  cnhubert路径

`-b`   bert路径

`-c`  1-5, 默认5，代表按标点符号切分。 1=凑四句一切 2=凑50字一切  3=按中文句号。切  4=按英文句号.切  5=按标点符号切



## API调用示例:

调用地址url: `http://你指定的ip:指定的端口`，默认是 `http://127.0.0.1:9880`


**调用时不指定参考音频**

启动 api2.py 时必须指定默认参考音频，才可在调用api时不指定，否则会失败:

GET方式调用，可直接浏览器中打开：

`http://127.0.0.1:9880?text=亲爱的朋友你好啊，希望你的每一天都充满快乐。&text_language=zh`

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dcc86a39958047fda782393c664729a0~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1137&h=713&s=31693&e=png&b=000000)


POST方式调用，以json格式传参:

```json
{
    "text": "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。",
    "text_language": "zh"
}
```

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/740bcf5d828c4a528273af2ff8d04471~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=877&h=356&s=31347&e=png&b=ffffff)



### 手动指定当次所使用的参考音频:

GET方式:

`http://127.0.0.1:9880?refer_wav_path=wavs/5.wav&prompt_text=为什么御弟哥哥，甘愿守孤灯。&prompt_language=zh&text=亲爱的朋友你好啊，希望你的每一天都充满快乐。&text_language=zh`

POST方式:

```json
{
    "refer_wav_path": "wavs/5.wav",
    "prompt_text": "为什么御弟哥哥，甘愿守孤灯。",
    "prompt_language": "zh",
    "text": "亲爱的朋友你好啊，希望你的每一天都充满快乐。",
    "text_language": "zh"
}
```

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1d715a9b3d1c4851a6dfd7f2e724d68b~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=835&h=447&s=39473&e=png&b=fffefe)



## Api调用返回信息:

成功时: 返回 wav 音频流，可直接播放或保存到 wav文件中，http 状态码 200

失败时: 返回包含错误信息的 json, http 状态码 400

```
{"code": 400, "message": "未指定参考音频且接口无预设"}
```





## 问题：想切换模型怎么办

api2.py和官方原版api.py 一样都不支持动态模型切换，也不建议这样做，因为动态启动加载模型很慢，而且在失败时也不方便处理。

**解决方法是：** 一个模型起一个api服务器，绑定不同的端口，在启动api2.py时，指定当前服务所要使用的模型和绑定的端口。

比如起2个服务，一个使用默认模型，绑定 9880 端口，一个绑定自己训练的模型，绑定 9881 端口，命令如下

**默认模型 9880 端口**: http://127.0.0.1:9880

`.\runtime\python api2.py -dr "5.wav" -dt "今天好开心" -dl zh `


**自己训练的模型**: http://127.0.0.1:9881

`.\runtime\python api2.py -p 9881  -s "SoVITS_weights/mymode-e200.pth" -g "GPT_weights/mymode-e200.ckpt" -dr "wavs/10.wav" -dt "御弟哥哥，为什么甘愿守孤灯" -dl zh `







