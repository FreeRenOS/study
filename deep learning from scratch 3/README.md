# 밑바닥부터 시작하는 딥러닝3
## 2021-03-04
- colab에서 MNIST train-images-idx3-ubyte.gz 다운로드시 HTTP 403 error 발행
- dezero/datasets.py 파일 github에서 다운로드 할 수 있게 변경
		
		 def prepare(self):
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}
		
		def prepare(self):
        ## 수정
        url = 'https://github.com/FreeRenOS/study/blob/main/deep%20learning%20from%20scratch%203/'
        # url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz?raw=true',
                       'label': 'train-labels-idx1-ubyte.gz?raw=true'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz?raw=true',
                      'label': 't10k-labels-idx1-ubyte.gz?raw=true'}
