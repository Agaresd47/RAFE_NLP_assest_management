你是总管的位置，所以你的context是最贵的，我要去睡觉了， 你自己跑，不需要像我确认任何事情， 你不要写任何代码， 你也不要读任何文件， 你的应该开不同agent去帮你读代码， 写代码， 跑试验，读文件， 实现代码。 

你能看到 我现在的 代码框架还行 也过得去， 我需要修正之前的错误， 你可以参考 /projects/p32908/nlp_code/another_codex_review.md （这个你自己读）来看别人的评价， 我们的目标是update我们的全套训练框架，能做两个事情：
1. Build non cot hf -> sft -> eval （这是之前的实现 我们不需要这个了， 我们只需要带RL的）
2. build cot hf -> sft cot style - > DPO base on stock price 

stock price return 在/projects/p32908/data/nlp 的三个csv里。 

你的工作模式： 读完了另一个gpt的 review和instruction （你不用全听 但是能帮你理解现在情况）
开子agent 简单的活开 5.4 low， 中等的开 mid， 如果是很复杂的玩意开high 来帮你读文件 总结 然后你得到信息a了就去开agent b 实行action b这样

我的需求： 明天早上醒来时 你应该build 好hf dataset了， 然后全部代码写好， 我开了gpu sesion就可以训练了。
如果你有问题 比如说两难 那么你可出两版 但我觉得应该没有

我的建议： 读json和csv时读前20行就够了， 别一个文件全cat
你让其他codex 链接tmux 去 ssh qnode0010

你干完活， 用 scancel 5397078  把这个cpu session停掉。 你现在正在登陆界面， 所以别跑任何程序，所有东西都用 这个cpu session去跑