!pip install torch pytorch_lightning lightning
import torch #untuk membuat tensors
import torch.nn as nn #untuk membuat neural network
import torch.nn.functional as f #untuk memberi akses ke activation dan loss function
from torch.optim import Adam #optimizer untuk weight dan bias

import lightning as L #untuk membuat neural networks menjadi lebih mudah
from torch.utils.data import TensorDataset, DataLoader #untuk training data
class LSTMbuatan(L.LightningModule):
    def __init__(self):
        super().__init__()
        ## Disini kita akan menginisialisasi nilai untuk Masing-masing Weight
        ## Disini kita bisa menggunakan 2 cara berbeda 1) Distribusi Normal 2) Distribusi Uniform
        ## Disini kita menggunakan Distribusi Normal
        ## Penjelasan "a brief norm dist" ada dibawah
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)        
        ## Membuat parameter dari weight pertama di LSTM
        ## Menggunakan torch.normal untuk menginisialisasi random number yang kita generated berdistribusi normal
        ## dengan mean=0 dan sd=1
        ## kita menggunakan requires_grad=TRUE untuk mengoptimasasi weight-weight ini.
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        ## disini kita membuat parameter untuk bias pertama. Disini kita menggunakan bias=0.
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        ## Begitu seterusnya
        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
        ## Kita bisa menginisialisasi weight dan bias dengan menggunakan distribusi uniform juga.
        ## seperti ini kira-kira bagaimana jika digunakan distribusi uniform
        #         self.wlr1 = nn.Parameter(torch.rand(1), requires_grad=True)
        #         self.wlr2 = nn.Parameter(torch.rand(1), requires_grad=True)
        #         self.blr1 = nn.Parameter(torch.rand(1), requires_grad=True)

        #         self.wpr1 = nn.Parameter(torch.rand(1), requires_grad=True)
        #         self.wpr2 = nn.Parameter(torch.rand(1), requires_grad=True)
        #         self.bpr1 = nn.Parameter(torch.rand(1), requires_grad=True)

        #         self.wp1 = nn.Parameter(torch.rand(1), requires_grad=True)
        #         self.wp2 = nn.Parameter(torch.rand(1), requires_grad=True)
        #         self.bp1 = nn.Parameter(torch.rand(1), requires_grad=True)
                
        #         self.wo1 = nn.Parameter(torch.rand(1), requires_grad=True)
        #         self.wo2 = nn.Parameter(torch.rand(1), requires_grad=True)
        #         self.bo1 = nn.Parameter(torch.rand(1), requires_grad=True)
    def lstm_unit(self, input_value, long_memory, short_memory):
        ## 1) The first stage yang menghitung persentase dari
        ##    seberapa banyak Long-Term Memory yang akan diingat LSTM
        ##    mekanismenya sama seperti yang dijelaskan di penjelasan sebelumnya.
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) + 
                                              (input_value * self.wlr2) + 
                                              self.blr1)
        
        ## 2) The second stage yang membuat new, potential long-term memory 
        ##    dan menghitung persentase dari seberapa banyak Long-Term Memory
        ##    yang akan ditambahkan
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) + 
                                                   (input_value * self.wpr2) + 
                                                   self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) + 
                                      (input_value * self.wp2) + 
                                      self.bp1)
        
        ## Disini kita mendapatkan update dari Long Term Memory
        updated_long_memory = ((long_memory * long_remember_percent) + 
                       (potential_remember_percent * potential_memory))
        
        ## 3) The third stage membuat new, potential short-term memory 
        ##    dan menghitung persentase dari seberapa banyak yang harus diingat
        ##    dan digunakan sebagai output.
        output_percent = torch.sigmoid((short_memory * self.wo1) + 
                                       (input_value * self.wo2) + 
                                       self.bo1)         
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent
        
        ## Disini kita mendapatkan long dan short term memories yang sudah terupdate.
        return([updated_long_memory, updated_short_memory])
    def forward(self, input): 
        
        long_memory = 0 
        short_memory = 0
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]
        
        ## Hari 1
        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        
        ## Hari 2
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        
        ## Hari 3
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        
        ## Hari 4
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)
        
        ##### return short_memory, Output dari LSTM
        return short_memory
    def configure_optimizers(self):
        return Adam(self.parameters())
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i)**2
        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)
            
        return loss
## Trying our homemade LSTM
model = LSTMbyHand() 

print("Sebelum di optimatisasi, parameternya adalah")
for name, param in model.named_parameters():
    print(name, param.data)

print("\nSekarang mari kita bandingkan nilai aktual dan nilai prediksinya")
## NOTE: Untuk prediksi ini, kita menggunakan data dari 4 hari harga saham dari masing-masing perusahaan. 
##       Yang berbeda dari nilai input dari data ini, hanya berbeda di hari pertama.
##       Hari pertama A memiliki nilai 0 dan hari pertama B memiliki nilai 1.
print("Company A: Nilai aktual = 0, Nilai Prediksi =", 
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Company B: Nilai aktual = 1, Nilai prediksi=", 
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
