if __name__ == '__main__':
    # %% [code] {"execution":{"iopub.status.busy":"2024-04-04T09:49:09.173660Z","iopub.execute_input":"2024-04-04T09:49:09.173935Z","iopub.status.idle":"2024-04-04T09:49:11.427213Z","shell.execute_reply.started":"2024-04-04T09:49:09.173913Z","shell.execute_reply":"2024-04-04T09:49:11.426418Z"}}
    import torch
    import torch.nn as nn
    import torch.utils.data
    import torch.nn.functional as F
    import torchtext
    import torchtext.data
    import math
    import numpy as np
    from tqdm import tqdm
    import time
    import random
    import matplotlib.pyplot as plt

    # INPUT_DATA_DIR = r'/kaggle/input/skypile-part/input_data.txt'
    # OUTPUT_DATA_DIR = r'/kaggle/input/skypile-part/output_data.txt'
    DATA_DIR = r'data.txt'

    EPOCHS = 100
    BATCH = 16

    VOCAB_SIZE = 100000
    DATASET_DOWNSAMPLE = int(1/0.025)
    LEARNING_RATE = 3e-4

    NUM_HEADS = 4
    DIM_HEAD = None # MHA内部会自动计算
    DIM_MODEL = 256
    DIM_FFN = DIM_MODEL*4
    NUM_LAYERS = 4
    ACTIVATION_FUNCTION = F.silu
    DROPOUT = 0.1
    # GRADIENT_ACCUMULATE = 10


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class Debug_Timer():
        def __init__(self) -> None:
            self.start_time = None
            self.name = None
        def timer_start(self, name=None):
            self.start_time = time.perf_counter()
            self.name = name
            print(f'{self.name}:', end='')

        def timer_stop(self):
            print(f"{round(time.perf_counter() - self.start_time, 4)}s")
        
    t = Debug_Timer()

    # %% [code] {"execution":{"iopub.status.busy":"2024-04-04T09:49:11.448710Z","iopub.execute_input":"2024-04-04T09:49:11.449032Z","iopub.status.idle":"2024-04-04T09:49:11.466729Z","shell.execute_reply.started":"2024-04-04T09:49:11.449007Z","shell.execute_reply":"2024-04-04T09:49:11.465895Z"}}
    class TextDatasetV3(torch.utils.data.Dataset):
        def tokenizer(self, x):
            return x.split(' ')
        def __init__(self, data_dir, vocab_size, downsample) -> None:
            """
            构造函数，初始化数据集对象
            参数:
                input_data_dir (str): 输入数据文件路径
                output_data_dir (str): 输出数据文件路径
            """
        
            self.word_counts = {}
            # self.tokenizer = tokenizer

            t.timer_start('读取数据')
            with open(data_dir, encoding='utf-8') as f:
                data = f.read().splitlines()  # 读取输入数据文件的每一行
            t.timer_stop()
            data = data[::downsample]
            
            # 将data每一句拆为input和output
            input_data = [self.tokenizer(line)[:-1] for line in data]
            output_data = [self.tokenizer(line)[1:] for line in data]
            
            self.input_data = input_data # 初始化
            self.output_data = output_data # 初始化
            
            print(self.input_data[0])
            print(self.output_data[0])

            t.timer_start('生成词频表')
            # 生成词频表
            for i in range(len(input_data)):
                input_line, output_line = input_data[i], output_data[i]
                word_list = list(set(input_line) | set(output_line))  # 将输入输出数据的词合并去重
                for word in word_list:
                    if word not in self.word_counts:
                        self.word_counts[word] = 1  # 统计词频

                    else:
                        self.word_counts[word] += 1  # 统计词频

            self.word_counts = {k: v for k, v in sorted(self.word_counts.items(), reverse=True, key=lambda item: item[1])}  # 将词频表按照频率从高到低排序
            t.timer_stop()
            self.word_counts = {key:self.word_counts[key] for key in list(self.word_counts.keys())[:vocab_size]} # 裁剪词典长度为vocab_size
            
            # 生成词表
            t.timer_start('生成词表')

            self.vocab = torchtext.vocab.vocab(self.word_counts , min_freq=1, specials=['<PAD>', '<OOV>'], special_first=True)
            
            self.index2word = self.vocab.get_itos()
            self.word2index = self.vocab.get_stoi()
            self.word_nums = len(self.index2word)

            self.vocab.set_default_index(self.vocab['<OOV>'])
            t.timer_stop()

            # 数据编码
            t.timer_start('数据编码')

            # 将文本数据转换为数字表示
            self.input_data = [torch.tensor([self.vocab[word] for word in sentence]) for sentence in tqdm(self.input_data)]
            # 输出数据暂时不转为one-hot 读取时转换
            # self.output_data = [F.one_hot(torch.tensor([self.vocab[word] for word in self.tokenizer(sentence)]),num_classes=self.word_nums) for sentence in tqdm(self.output_data)]
            self.output_data = [torch.tensor([self.vocab[word] for word in sentence]) for sentence in tqdm(self.output_data)]
            print(self.input_data[0])
            print(self.output_data[0])
            # 数据padding
            self.input_data = nn.utils.rnn.pad_sequence(self.input_data, batch_first=True)
            self.output_data = nn.utils.rnn.pad_sequence(self.output_data, batch_first=True)
            self.seq_max_len = len(self.input_data[0]) # padding后元素一样长
            t.timer_stop()
        
        def __len__(self):
            return len(self.input_data)

        def __getitem__(self, index):
            """
            根据索引获取数据样本
            参数:
                index (int): 索引值
            返回值:
                tuple: 输入数据和输出数据的元组，以tensor形式表示
            """
            return (self.input_data[index].clone().detach(), self.output_data[index].clone().detach().long())
        

    # %% [code] {"execution":{"iopub.status.busy":"2024-04-04T09:49:11.467450Z","iopub.execute_input":"2024-04-04T09:49:11.467691Z","iopub.status.idle":"2024-04-04T09:49:13.928831Z","shell.execute_reply.started":"2024-04-04T09:49:11.467670Z","shell.execute_reply":"2024-04-04T09:49:13.927851Z"}}
    dataset = TextDatasetV3(DATA_DIR, vocab_size=VOCAB_SIZE, downsample=DATASET_DOWNSAMPLE)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)

    # %% [code] {"execution":{"iopub.status.busy":"2024-04-04T09:49:13.930016Z","iopub.execute_input":"2024-04-04T09:49:13.930318Z","iopub.status.idle":"2024-04-04T09:49:13.946482Z","shell.execute_reply.started":"2024-04-04T09:49:13.930292Z","shell.execute_reply":"2024-04-04T09:49:13.945623Z"}}
    class CustomDecoderLayer(nn.Module):
            # CustomTransformerDecoderLayer
            def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1, activation = F.relu, batch_first=True):
                super().__init__()
                # 使用GPT2结构
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.mutihead_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=batch_first)
                self.linear1 = nn.Linear(d_model, d_ffn, bias=True)
                self.dropout = nn.Dropout(dropout)
                self.linear2 = nn.Linear(d_ffn, d_model, bias=True
                                        )
                self.activation = activation
                # self.dropout1 = nn.Dropout(dropout)
                self.dropout2 = nn.Dropout(dropout)
            
            def forward(self, x, attn_mask, key_padding_mask, is_causal):
                res = x
                x = self.norm1(x)
                x = self.mutihead_self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal)[0]
                x = res + x
                res = x
                x = self.norm2(x)
                x = res + self._ffn(x)
                return x

            def _ffn(self, x):
                x = self.activation(self.linear1(x))
                x = self.dropout(x)
                x = self.linear2(x)
                return self.dropout2(x)

    class CustomGPT(nn.Module):
        def __init__(self,vocab_size, seq_max_len, total_word, d_model, d_ffn, n_heads, d_head, n_layers, dropout_rate, activation, device):        
            super().__init__()  # 调用父类的初始化方法
            # self.vocab_size = vocab_size  # 词汇表大小
            # self.d_ffn = d_ffn  # FFN嵌入维度
            # self.n_heads = n_heads  # 头数数
            # self.d_head = d_head  # 头嵌入维度
            # self.activation = activation
            # self.dropout_rate = dropout_rate  # Dropout率
            self.n_layers = n_layers  # 层数
            self.total_word = total_word  # 总词数
            self.d_model = d_model  # 模型嵌入维度
            self.seq_max_len = seq_max_len  # 序列最大长度
            self.device = device
            self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)  # Token嵌入层
            self.pos_emb = nn.Embedding(num_embeddings=seq_max_len, embedding_dim=d_model)  # 位置嵌入层
            self.decoder_layer = CustomDecoderLayer(d_model=d_model, d_ffn=d_ffn, n_heads=n_heads, dropout=dropout_rate, activation=activation)
            self.out_fc = nn.Linear(self.d_model, self.total_word, bias=True)  # 输出全连接层 
            self.last_norm = nn.LayerNorm(d_model)

        def forward(self, x):
            pad_mask = x==0
            pos = torch.arange(self.seq_max_len).to(self.device)
            emb = self.tok_emb(x) * math.sqrt(self.d_model) + self.pos_emb(pos)
    #         causal_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_max_len).to(self.device)
            
            decoder_list = nn.ModuleList([self.decoder_layer for _ in range(self.n_layers)])
            x = emb
            for layer in decoder_list:
                x = layer(x, attn_mask=None, key_padding_mask=pad_mask, is_causal=True)
            # 使用is_causal无需attn_mask
            x = self.last_norm(x)
            out = self.out_fc(x) # CrossEntropyLoss输出时不需要softmax
            return out
            

    # %% [code] {"execution":{"iopub.status.busy":"2024-04-04T09:49:13.947591Z","iopub.execute_input":"2024-04-04T09:49:13.947882Z","iopub.status.idle":"2024-04-04T09:49:13.961374Z","shell.execute_reply.started":"2024-04-04T09:49:13.947848Z","shell.execute_reply":"2024-04-04T09:49:13.960610Z"}}
    class TextGenerator():
        def __init__(self, model: nn.Module, dataset: torchtext.vocab.Vocab) -> None:
            self.model = model
            self.dataset = dataset

        def generate(self, start_token: list, seq_len=10, temperature=1, print_out=True):
            with torch.no_grad():
                self.model.eval()
                tok = start_token.copy()
                for i in range(seq_len):
                    # print(tok)
                    seq = [self.dataset.vocab[j] for j in tok]
    #                 print(seq)
                    seq = nn.utils.rnn.pad_sequence([torch.tensor(seq).clone().detach().to(device), torch.zeros(self.dataset.seq_max_len*torch.cuda.device_count())], batch_first=True)[0] # 加一项保证padding长度为seq_max_len
                    out = self.model(seq.long().to(device))
                    out /= temperature
                    out = F.softmax(out, dim=-1)
                    out_word = self.dataset.vocab.get_itos()[out[len(tok)-1].multinomial(num_samples=1)]
    #                 print([self.dataset.vocab.get_itos()[out[len(tok)+i].argmax()] for i in range(-2, 0)])
    #                 print(out[len(tok)-1].max(), end='  ')
    #                 print(out[len(tok)-1].argmax(), end='\n\n')
                    if print_out:
                        print(out_word, end=' ')
                    tok.append(out_word)

            return tok
                    

    # %% [code] {"execution":{"iopub.status.busy":"2024-04-04T09:49:13.962561Z","iopub.execute_input":"2024-04-04T09:49:13.963176Z","iopub.status.idle":"2024-04-04T09:49:15.184236Z","shell.execute_reply.started":"2024-04-04T09:49:13.963143Z","shell.execute_reply":"2024-04-04T09:49:15.183176Z"}}

    # print(dataset.word2index)
    # print([dataset[i] for i in range(len(dataset))])
    print(f'词数: {dataset.word_nums}')
    dataset_len = len(dataset)
    print(f'数据集数量：{dataset_len}')
    model = CustomGPT(
        vocab_size=dataset.word_nums,
        seq_max_len=dataset.seq_max_len,
        total_word=dataset.word_nums,
        d_model=DIM_MODEL,
        d_ffn=DIM_FFN,
        n_heads=NUM_HEADS,
        d_head=DIM_HEAD,
        n_layers=NUM_LAYERS,
        dropout_rate=DROPOUT,
        activation=ACTIVATION_FUNCTION,
        device=device
        )

    # model.half()
    model.to(device)
    decoder_params = sum(p.nelement() for p in model.decoder_layer.parameters()) * (NUM_LAYERS - 1) # decoderlayer堆叠导致参数计算错误
    total_params = sum(p.nelement() for p in model.parameters()) + decoder_params

    # torch.compile(model)
    print(f'模型参数量: {total_params/1e6}M')
    print('训练')
    print(dataset.index2word[1:30])
    # print([(i, item) for i, item in enumerate(dataset.index2word)])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=6, T_mult=2, eta_min=6e-5, last_epoch=-1, verbose=True)
    loss_log = []

    # %% [code] {"execution":{"iopub.status.busy":"2024-04-04T09:49:15.185673Z","iopub.execute_input":"2024-04-04T09:49:15.186324Z"}}
    # 训练模型
    for epoch in range(EPOCHS):
    #     loss_log = []
        loss_sum = 0
        bar = tqdm(dataloader, unit='batch')
        torch.cuda.empty_cache()
        for i, (inputs, targets) in enumerate(dataloader):
            # print(f'Step: {i}')
            model.train()
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()  
            output = model(torch.as_tensor(inputs))
    #         print(inputs.size(), output.size(), targets.size())

            loss = criterion(output.view(-1, dataset.word_nums), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            loss_log.append(loss.item())
            loss_sum += loss.item()


            if i % 20 == 0:
                start = [random.choice(['LLMsystemNEWLINE', '水', '我', '出'])]
                generator = TextGenerator(model, dataset)
                T = random.uniform(0.1, 1)
                ans = generator.generate(start_token=start, seq_len=10, temperature=T, print_out=False)
                ans = ans[len(start):] # 截掉start_token
                print(f'Step: {i} Temperature: {T} (input){start}->', end='')             
                for i in ans:
                    print(i, end=' ')
                print()


            bar.update(1)
            bar.postfix = f'Loss: {round(loss.item(), 5)}'
        # scheduler.step()
        bar.close()
        
        
        with torch.no_grad():
            print()
            print(f"Epoch: {epoch+1}, Loss_sum: {loss_sum}, Loss_avg: {loss_sum/len(dataloader)}")
            if epoch % 1 == 0:
                start = [random.choice(dataset.index2word[2:20])] # 取词表中词频最高的20词(不含OOV PAD)
                generator = TextGenerator(model, dataset)
                ans = generator.generate(start_token=start, seq_len=20, print_out=False)
                ans = ans[len(start):] # 截掉start_token
                print(f'Step: {i} (input){start}->', end='')
                for i in ans:
                    print(i, end=' ')
                print()
            
            if epoch % 1 == 0:
                plt.subplots(figsize=(15, 4))
                plt.subplot(1, 2, 1)
                plt.plot(loss_log)
                plt.yscale('log')  # 将 y 轴设置为对数坐标轴
                plt.xlabel('Iteration')
                plt.ylabel('Loss(log)')

                plt.subplot(1, 2, 2)
                plt.plot(loss_log)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
        #             display.clear_output(wait=True)
                plt.pause(0.00000001) 
                

    # %% [code]


    # %% [code]
    # !pip install ipympl --upgrade

    # %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false}}
    test_generator = TextGenerator(model, dataset)
    MAX_LEN = 40
    while True:
        start = input('In>>')
        print(*test_generator.generate(start_token=start.split(' '), seq_len=MAX_LEN, temperature=1, print_out=False))