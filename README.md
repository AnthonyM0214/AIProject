# 自然语言处理——大众点评星级预测
---
## 目的
## 数据集简介
来源：http://opendata.pku.edu.cn/dataset.xhtml?persistentId=doi:10.18170/DVN/GCIUN4&version=1.0
* 数据描述：	一段时间内，大众点评上广州地区粤菜排名前211家餐厅的用户评论信息，包括评论用户、评论时间、评论内容、点赞数、回复数。
* 时间范围：2004-7-1 至2017-10-31
* 数据量：467455 
* 数据格式：csv
* 数据字段说明：
* 
|字段名称  | 字段类型 |字段描述|
|--|--|--|		
|Review_ID|	long|	评论id
|Merchant|	string|	评论餐厅名称
Rating|	int|	餐厅整体评分
Score_taste|	int|	味道评分
Score_environment|	int|	环境评分
Score_service|	int|	服务评分
Price_per_person|	int|	人均价格（Null为空）
Time	|time|	评论时间
Num_thumbs_up|	int|	评论点赞数
Num_ response|	int|	评论回复数
Content_review	|string|	评论文本
Reviewer|	string|	评论人用户名
Reviewer_value|	int|	评论人等级
Reviewer_rank|	int|	评论人是否为VIP用户（1为是，0为否）
Favorite_foods	|string|	喜欢的菜

### 样例数据![样例数据](https://img-blog.csdnimg.cn/20190702211737886.png)

---
## 数据预处理
### 读取数据集
截取Rating和Content_review的部分。因为数据集是取排名靠前的餐厅，所以从分布上来看好评远远大于差评。大众点评的星级一共有五星，从五星到一星好评程度递减。由pandas自带的describe函数统计出40万条评论中只有一万条左右是一到二星的，所以最后每个星级采了10000个样本构成一个新的数据集。

![读取数据集](https://img-blog.csdnimg.cn/20190702211832352.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FudGhvbnlNMDg=,size_16,color_FFFFFF,t_70)

### 分词处理
由于中文编码问题，首先要转换中文编码，然后使用结巴分词，将数据集分为两列，第一列为星级，第二列为以空格分词的文本串。

![分词](https://img-blog.csdnimg.cn/20190702211912673.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FudGhvbnlNMDg=,size_16,color_FFFFFF,t_70)
## 搭建模型
### RNN
RNN的假设——事物的发展是按照时间序列展开的，即前一刻发生的事物会对未来的事情的发展产生影响。所以，在处理过程中，每一刻的输出都是带着之前输出值加权之后的结果。当前的结果包含之前的结果，或者说受到之前结果的影响。也就是说是基于句子上下文做出的推断。
嵌入层+双向RNN+最大池化层+全连接层+5%的遗忘层+全连接层
```
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```
### CNN
CNN神经网络是模仿人类处理信息的过程，CNN神经网络，有一个基础的假设——人类的视觉总是会关注视线内特征最明显的点，首先生成一个滤镜，并对图像整体进行扫描过滤，通过这个滤镜filter解析，得到很多个扫描后的图片分支结果。
提取每个小特征当中，值最大的那个。（值越大说明特征越明显，越符合上文说的人眼特性）
通过不停的特征抽取，得到最后的结果，如果这个结果与我们的预期不符，则计算误差值，反馈给每一层的卷积网络，进行微调整，再重复上面的步骤。

```
model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

## 结果
### RNN
![RNN](https://img-blog.csdnimg.cn/20190702213251845.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FudGhvbnlNMDg=,size_16,color_FFFFFF,t_70)
### CNN
![CNN](https://img-blog.csdnimg.cn/20190702213225422.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FudGhvbnlNMDg=,size_16,color_FFFFFF,t_70)
CNN 是分层架构，RNN 是连续结构。在处理语言的任务上，基于它们的特性“分层（CNN） vs. 连续（RNN）”，我们倾向于为分类类型的任务选择 CNN，例如情感分类，因为情感通常是由一些关键词来决定的；对于顺序建模任务，我们会选择 RNN，例如语言建模任务，要求在了解上下文的基础上灵活建模。模型训练效果很不理想最后准确率只有百分之20左右，数据集中的文本其实有很多信息都是噪声，很多人喜欢在评论里报流水账，但可能和最终的Rating没有很大关系，而大部分关系到星级的会直接表达“好”或者是“差”。
虽然CNN在数据集上训练的很好，但是在测试集的上有过拟合现象。首先是数据量相对于特征参数来说还是不够，可能因为美食评论还是比较个性化的。还有虽然在模型中添加了说了文本中无关的噪声比较多，所以效果不佳。



