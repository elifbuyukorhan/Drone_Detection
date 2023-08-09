# Giriş
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