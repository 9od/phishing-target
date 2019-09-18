## 图像相似度比较

> 接在 YOLO 模型后
>
> 通过比较钓鱼网站 logo 与 TargetList 白名单中logo的图像相似度找到钓鱼网站的钓鱼对象  

* GoogleSearch : 调用 GoogleImg 网页对YOLO得到的网页logo图形进行识图，得到钓鱼对象。
* OCR :  图像转文字。
* Siamese Network : 孪生神经网络。
* Traditional Methods : 传统cv领域方法