# Drone Detection 

## Giriş
Aşağıdaki adımlar bir Docker kapsayıcısında (container) GPU destekli Tensorflow kullanımı, bir Docker imajı (image) oluşturma, kapsayıcı (container) içerisinde bir IDE çalıştırma konularında notlar içermektedir.

## Docker'ı Yükleyin

```
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

Test with:
```
sudo docker run hello-world
```

Sudo ile komut çalıştırmak zorunda kalmamak için kullanıcı kimliğinizi docker grubuna ekleyin:
```
sudo usermod -aG docker $USER
newgrp docker
```
Doğrulamak için:
```
docker run hello-world
```

## NVIDIA Container Toolkit ve Tensorflow'u Yükleyin
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
Test etmek için:
```
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
docker run --gpus all --rm nvidia/cuda nvidia-smi
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
Not: Belgeler, 
```
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))
```
çalıştırmayı söylüyor  ancak AttributeError'a göre: 'tensorflow' modülünün TensorFlow 2'de 'enable_eager_execution' özelliği yok, bu bir hata veriyor, bu yüzden yukarıdaki komutu kullanın.

Hangi TensorFlow sürümünün kurulu olduğunu doğrulamak için:
```
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu python -c 'import tensorflow as tf; print(tf.__version__)'
```

## Tensorflow ve Docker'ı Kullanma

Geliştirme ve test etme amaçları için, git tarafından yönetilen python ve kaynak dosyaları okuyabileceği ve eğitilmiş modeli, kapsayıcı (container) durdurulduğunda devam edecek bir yere kaydedebileceği bir çalışma dizinine bağlanan bir kapsayıcıyı (container) basitçe başlatabilirsiniz.

Geliştirici oturumları şu yollarla başlatılır:
```
docker run --gpus all -it -u 1000:1000 -p 8888:8888 --mount type=bind,src=/home/<username>/projects,dst=/home/projects --env HOME=/home/projects -w /home/projects tensorflow/tensorflow:latest-gpu-py3 bash
```

* 1000 ve 1000, çalıştırmak istediğim grup kimlikleri olan kullanıcıdır. (böylece container tarafından yazılan dosyalar container dışında doğru izinlere sahip olur.)
* /home/\<username>/projects gerçek dosya sistemidir ve /home/projects sanal olanı (böylece yerel git deposuna erişimim var, bu kapsayıcıda (container) git'in yüklü olmadığına dikkat edin, bu nedenle git pull/push vb. kapsayıcı (container) dışında gerçekleştirilecek.)
* $HOME ortam değişkeni, varsayılan yerine /home/projects olarak ayarlanmıştır. (bu, Visual Studio Code Remote - Containers uzantısı içindir.)
* Çalışma dizini /home/projects olarak ayarlanmıştır.
* GPU destekli Tensorflow image ve Python 3 (tensorflow/tensorflow:latest-gpu-py3) kullanılır.
* Bir bash shell başlatılır.

## Bir Image Oluşturmak

Ek paketler yüklemek istiyorsanız, bir image oluşturmak daha kolay olacaktır. Ben Jupyter Notebook'u kurmak istiyorum, bu yüzden aşağıdaki gibi temel bir Docker dosyası oluşturacağım:

```
FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /home/projects
RUN pip install notebook
``` 
Çalıştırmak için:
```
docker build --tag tfdev .
```
Ve sonraki oturumları orijinal komutun basitleştirilmiş bir versiyonuyla başlatın, yani:
```
docker run --gpus all -it -u 1000:1000 -p 8888:8888 --mount type=bind,src=/home/<username>/projects,dst=/home/projects --env HOME=/home/projects tfdev bash
```
Elbette bu, doğrudan kaynağı okuyabilen ve yazabilen bir kullanıcı olarak etkileşimli olarak çalıştırmaya devam ediyor. Geliştirme aşamasından üretim aşamasına geçebileceğiniz tamamen bağımsız bir image oluşturmak istiyorsanız, kaynak kodunu kopyalamak gibi ek adımlara ihtiyacınız olacaktır.

## Tensorflow Docker Image'dan Visual Studio Code Çalıştırmak

Bu, kod tamamlama vb.'nin IDE'de çalışması içindir. Visual Studio Code kullanıyorum, ancak muhtemelen diğer IDE'ler için benzer yaklaşımlar var. Visual Studio Code, sırasıyla Docker Compose'a ihtiyaç duyan Remote - Containers uzantısını kullanır.

Docker Compose yüklemek için:
```
sudo curl -L "https://github.com/docker/compose/releases/download/1.25.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```
Visual Studio Code Uzaktan Geliştirme Uzantı Paketi'ni yüklemek için Git (Go) > Dosyaya Git'e (Go to File) gidin ve şunu girin:
```
ext install ms-vscode-remote.vscode-remote-extensionpack
```
Kurulduğunda ve bir docker çalışırken, Visual Studio Kod Etkinlik Çubuğunda Uzak Gezgin'e gidin, çalışmakta olan container'ı seçin, sağ tıklayın ve "Kapsayıcıya iliştirin" ("Attach to container"). –env HOME, orijinal docker run komutunda (veya eşdeğerinde) ayarlanmazsa, /root/ içinde .vscode-server oluşturmaya çalışacaktır ve yetersiz izinler nedeniyle "Command in container failed: mkdir -p /root/.vscode-server/” hatası verecektir.

## Jupyter Not Defterlerini Docker içinden Çalıştırma

Varsayılan tensorflow/tensorflow:latest-gpu image'ı içinde bir pip listesi yaparsanız, Jupyter Notebooks'un varsayılan olarak yüklenmediğini göreceksiniz. Yukarıdaki "Image oluşturma" bölümüne göre özel bir Docker image ile kurulabilir.

Jupyter Notebook'u etkileşimli Docker kapsayıcısında başlatmak için:
```
jupyter notebook --ip 0.0.0.0 --no-browser
```
–ip yerel ana bilgisayarı belirtir ve –no-browser ona normalde yaptığı gibi bir tarayıcı oturumu başlatmamasını söyler. Jüpyter not defteri işlemi başladığında gösterilen belirteci girmenizin isteneceğini belirterek, http://localhost:8888 aracılığıyla kapsayıcının dışına erişebilirsiniz.

## Docker Image'larını en son Tensorflow'u kullanacak şekilde güncelleme
Yukarıdakilerin hepsini ayarlayıp daha sonra tekrar ziyaret ederseniz, image'ları Tensorflow'un sonraki sürümlerini kullanmak için güncellemeniz gerektiğini fark edebilirsiniz. Bu, şu yollarla yapılabilir:
```
docker pull tensorflow/tensorflow:latest-gpu
```
ve/veya
```
docker pull tensorflow/tensorflow:latest-gpu-py3
```

İçeriği hazırladığım [linke](https://michael-lewis.com/posts/setting-up-tensorflow-with-docker-and-nvidia-container-toolkit/) buradan ulaşabilirsiniz.  

<br>
<br>

## Bilmeniz Gereken Temel Docker Terimleri

* **Image:** Docker image'ları, container'ların temelidir. Image'lar, bir container çalışma zamanında kullanılmak üzere kök dosya sistemi değişikliklerinin ve karşılık gelen yürütme parametrelerinin değişmez, sıralı bir koleksiyonudur.
* **Container:**  Bağımsız bir uygulama olarak yürütülebilen image'ın bir örneğidir. Container, ana makineden izole edilmiş bağımlılıklar olan image'ın üstünde yer alan değişken bir katmana sahiptir.
* **Volume:** Volume, Birlik Dosya Sistemini (Union File System) atlayan bir veya daha fazla kapsayıcı içinde özel olarak belirlenmiş bir dizindir. Volumes, container'ın yaşam döngüsünden bağımsız olarak verileri kalıcı kılmak için tasarlanmıştır.
* **Registery:** Docker image'larını dağıtmak için kullanılan bir depolama ve içerik dağıtım sistemi.
* **Repository:** Genellikle aynı uygulamanın farklı sürümleri olan ilgili Docker image'larından oluşan bir koleksiyon.

Docker komutları (Docker Cheat Sheet) hakkında daha fazla bilgi almak için [linki](https://www.exxactcorp.com/blog/Deep-Learning/docker-cheat-sheet-for-deep-learning-the-basics) ziyaret edebilirsiniz. 

<br>
<br>

# Temelleri Öğrenelim
Çoğu makine öğrenimi (machine learning) iş akışı, verilerle çalışmayı, modeller oluşturmayı, model parametrelerini optimize etmeyi ve eğitilmiş modelleri kaydetmeyi içerir. Bu eğitimde, bu kavramların her biri hakkında daha fazla bilgi edinmek için bağlantılar içeren, PyTorch'ta uygulanan bir makine öğrenimi iş akışı tanıtılmaktadır.

Bir girdi görüntüsünün şu sınıflardan birine ait olup olmadığını tahmin eden bir sinir ağını eğitmek için FashionMNIST veri kümesini kullanacağız: Tişört/üst, Pantolon, Kazak, Elbise, Ceket, Sandalet, Gömlek, Spor Ayakkabı, Çanta veya Ayak Bileği bot. (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, or Ankle boot.)

<br>

## Step 0 
Bu bölüm, makine öğrenimindeki yaygın görevler için API üzerinden çalışır. 

### Verilerle çalışma
PyTorch'un verilerle çalışmak için iki ilkesi vardır: `torch.utils.data.DataLoader` ve `torch.utils.data.Dataset`. Veri kümesi, örnekleri ve bunlara karşılık gelen etiketleri depolar ve `DataLoader`, `Dataset` etrafına bir yinelemeyi sarar.

``` 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

PyTorch, tümü veri kümeleri içeren  [TorchText](https://pytorch.org/text/stable/index.html), [TorchVision](https://pytorch.org/vision/stable/index.html) ve [TorchAudio](https://pytorch.org/audio/stable/index.html) gibi alana özgü kitaplıklar sunar. Bu eğitim için bir TorchVision veri seti kullanacağız.

Torchvision.datasets modülü, CIFAR, COCO gibi birçok gerçek dünya görüş verisi için Veri Kümesi nesneleri içerir ([tam liste burada](https://pytorch.org/vision/stable/datasets.html)). Bu öğreticide, FashionMNIST veri kümesini kullanıyoruz. Her TorchVision Veri Kümesi iki argüman içerir: sırasıyla örnekleri ve etiketleri değiştirmek için `transform` ve `target_transform`.

```
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```
`Dataset`'i `DataLoader`'a bir argüman olarak iletiyoruz. Bu, veri setimiz üzerinde yinelemeyi tamamlar ve otomatik gruplamayı, örneklemeyi, karıştırmayı ve çok işlemli veri yüklemeyi destekler. Burada 64'lük bir toplu iş (batch) boyutu tanımlıyoruz, yani yinelenebilir dataloader'daki her öğe, 64 özellik ve etiketten oluşan bir toplu iş (batch) döndürecektir.

```
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```
Out:
```
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```
<br>

### Model Yaratma

PyTorch'ta bir neural network tanımlamak için [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)'den miras alan (inherit eden) bir sınıf oluşturuyoruz. `__init__` fonksiyonunda ağın katmanlarını tanımlıyoruz ve forward fonksiyonunda verilerin ağ üzerinden nasıl geçeceğini belirtiyoruz. Neural network'teki işlemleri hızlandırmak için varsa GPU'ya veya MPS'ye taşıyoruz.
```
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```
Out: 
```
Using cuda device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```
<br>

### Model Parametrelerini Optimize Etme

Bir modeli eğitmek için bir kayıp fonksiyonuna ([loss function](https://pytorch.org/docs/stable/nn.html#loss-functions)) ve bir optimize ediciye ([optimizer](https://pytorch.org/docs/stable/optim.html)) ihtiyacımız var.

```
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```
Tek bir eğitim döngüsünde model, eğitim veri kümesi (training dataset) üzerinde tahminler yapar (fed to it in batches) ve modelin parametrelerini ayarlamak için tahmin hatasını geri yayar (backpropagates).

```
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

Ayrıca öğrendiğinden emin olmak için modelin performansını test veri kümesiyle (test dataset) karşılaştırırız.

```
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

```
Eğitim süreci birkaç yineleme (epochs) üzerinden yürütülür. Her epoch, model daha iyi tahminler yapmak için parametreleri öğrenir. Modelin doğruluğunu (accuracy) ve kaybını (loss) her epoch'ta yazdırıyoruz; her epoch'ta doğruluğun arttığını ve kaybın azaldığını görmek isteriz.

```
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```
Out: 
```
Epoch 1
-------------------------------
loss: 2.312952  [   64/60000]
loss: 2.304456  [ 6464/60000]
loss: 2.278305  [12864/60000]
loss: 2.271326  [19264/60000]
loss: 2.254906  [25664/60000]
loss: 2.229268  [32064/60000]
loss: 2.241202  [38464/60000]
loss: 2.209241  [44864/60000]
loss: 2.213543  [51264/60000]
loss: 2.183385  [57664/60000]
Test Error: 
 Accuracy: 32.7%, Avg loss: 2.171360 

Epoch 2
-------------------------------
loss: 2.186638  [   64/60000]
loss: 2.178233  [ 6464/60000]
loss: 2.120461  [12864/60000]
loss: 2.136642  [19264/60000]
loss: 2.094708  [25664/60000]
loss: 2.043836  [32064/60000]
loss: 2.070754  [38464/60000]
loss: 2.004480  [44864/60000]
loss: 2.018052  [51264/60000]
loss: 1.946542  [57664/60000]
Test Error: 
 Accuracy: 57.0%, Avg loss: 1.933942 

Epoch 3
-------------------------------
loss: 1.969325  [   64/60000]
loss: 1.937948  [ 6464/60000]
loss: 1.824028  [12864/60000]
loss: 1.859098  [19264/60000]
loss: 1.756827  [25664/60000]
loss: 1.711751  [32064/60000]
loss: 1.721016  [38464/60000]
loss: 1.631972  [44864/60000]
loss: 1.656443  [51264/60000]
loss: 1.542573  [57664/60000]
Test Error: 
 Accuracy: 60.7%, Avg loss: 1.552148 

Epoch 4
-------------------------------
loss: 1.621543  [   64/60000]
loss: 1.578785  [ 6464/60000]
loss: 1.428637  [12864/60000]
loss: 1.492114  [19264/60000]
loss: 1.374401  [25664/60000]
loss: 1.375991  [32064/60000]
loss: 1.372361  [38464/60000]
loss: 1.311350  [44864/60000]
loss: 1.351148  [51264/60000]
loss: 1.236402  [57664/60000]
Test Error: 
 Accuracy: 62.6%, Avg loss: 1.262719 

Epoch 5
-------------------------------
loss: 1.347239  [   64/60000]
loss: 1.319678  [ 6464/60000]
loss: 1.157217  [12864/60000]
loss: 1.256537  [19264/60000]
loss: 1.135083  [25664/60000]
loss: 1.166265  [32064/60000]
loss: 1.171427  [38464/60000]
loss: 1.124478  [44864/60000]
loss: 1.170500  [51264/60000]
loss: 1.071890  [57664/60000]
Test Error: 
 Accuracy: 64.0%, Avg loss: 1.094037 

Done!
```
### Modelleri Kaydetme

Bir modeli kaydetmenin yaygın bir yolu, dahili durum sözlüğünü (internal state dictionary) (model parametrelerini içeren) seri hale getirmektir.

```
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
``` 

Out:
```
Saved PyTorch Model State to model.pth
```

### Modelleri Yükleme
Bir modeli yükleme süreci, model yapısının yeniden oluşturulmasını ve buna durum sözlüğünün (state dictionary) yüklenmesini içerir.

```
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
```

Bu model artık tahminlerde bulunmak için kullanılabilir.
```
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
``` 
Out:
```
Predicted: "Ankle boot", Actual: "Ankle boot"
``` 

<br>

step0_starting.py dosyası yukarıdaki aşamaların hepsini içermektedir. Tutorial1 dosyası içerisinden ulaşabilirsiniz.

İlerleyen adımların detayları linkler aracılığıyla belirtilecektir. İlgili linke tıklayarak kod hakkında daha fazla bilgiye sahip olabilirsiniz. Bu aşama için [linke](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) tıklayabilirsiniz. 

<br>

## Step 1

### Tensors

Tensors, dizilere ve matrislere çok benzeyen özel bir veri yapısıdır. PyTorch'ta, bir modelin giriş ve çıkışlarının yanı sıra modelin parametrelerini kodlamak için tensors kullanırız.

Tensors, GPU'larda veya diğer donanım hızlandırıcılarda çalışabilmesi dışında [NumPy](https://numpy.org/)'nin ndarray'lerine benzer. Aslında, tensors ve NumPy dizileri genellikle aynı temel belleği paylaşarak veri kopyalama ihtiyacını ortadan kaldırır (bkz. NumPy ile Köprü). Tensors ayrıca otomatik farklılaşma için optimize edilmiştir. ndarray'lere aşina iseniz, Tensor API ile kendinizi rahat hissedeceksiniz.

Tutorial1 klasörü içerisindeki step1_tensors.py kodu çalıştırıldığında aşağıdaki çıktıyı vermektedir. Kod içindeki aşamaları [linke](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) tıklayarak takip edebilirsiniz. 

```
tensor([[1, 2],
        [3, 4]])
tensor([[1, 2],
        [3, 4]])
Ones Tensor: 
 tensor([[1, 1],
        [1, 1]]) 

Random Tensor: 
 tensor([[0.2809, 0.5341],
        [0.9029, 0.8020]]) 

Random Tensor: 
 tensor([[0.5650, 0.6653, 0.4215],
        [0.8167, 0.7467, 0.6372]]) 

Ones Tensor: 
 tensor([[1., 1., 1.],
        [1., 1., 1.]]) 

Zeros Tensor: 
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
Tensor: tensor([[0.2975, 0.6128, 0.6737, 0.7769],
        [0.6401, 0.1322, 0.8304, 0.2493],
        [0.5964, 0.0468, 0.6274, 0.6829]])
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
12.0 <class 'float'>
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]]) 

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```

## Step 2 
### DATASETS & DATALOADERS

Veri örneklerinin işlenmesine yönelik kod dağınık hale gelebilir ve bakımı zorlaşabilir; İdeal olarak, daha iyi okunabilirlik ve modülerlik için veri kümesi kodumuzun model eğitim kodumuzdan ayrılmasını istiyoruz. PyTorch iki veri ilkesi sağlar: önceden yüklenmiş veri kümelerini ve kendi verilerinizi kullanmanıza izin veren `torch.utils.data.DataLoader` ve `torch.utils.data.Dataset`. `Dataset`, örnekleri ve bunlara karşılık gelen etiketleri depolar ve `DataLoader`, örneklere kolay erişim sağlamak için `Dataset` etrafına bir yineleme sarar.

PyTorch domain kütüphaneleri, `torch.utils.data.Dataset` alt sınıfını oluşturan ve belirli verilere özgü işlevleri uygulayan bir dizi önceden yüklenmiş veri kümesi (ModaMNIST gibi) sağlar. Modelinizi prototiplemek ve kıyaslamak için kullanılabilirler. 

Tutorial1 klasörü içerisindeki step2_datasets&dataloaders.py kodu çalıştırıldığında aşağıdaki çıktıyı ve aynı klasör içindeki my_plot1.png ve my_plot2.png resimlerini vermektedir. Kod içindeki aşamaları [linke](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) tıklayarak takip edebilirsiniz. 

```
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 1
```
