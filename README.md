# Color-Spatial-AutoAugment
The Implementation used to test Color-Spatial AutoAugment in "Color-Spatial AutoAugment Another Approach To AutoAugment Policies Found &amp; Implementation".

Pretrained Models and their TensorBoard can be found [Here](https://drive.google.com/drive/folders/1-0jRnV04WfVQ9Qz-ZN_aOPneiljq6UW7?usp=sharing).

**Required Packages**
* Torch Package.
* TorchSummary
* tqdm
* scipy
* scikit-learn
* seaborn
* tensorboard

install the required packages using 

```
pip install -r requirements.txt
```

Or run ```venv_Install.ps1``` to create the `venv` and install the required packages.

**Training Steps**
* Run the venv or run `venv_Run.ps1` to activate the `venv`.
* run `main.py` to start the training or  edit `config.py` for extra settings.
* `--augment` have two options `AutoAugment` to train using *AutoAugment* and `Augment` to run using *Color-Spatial AutoAugment*.
* `--dataset` either `cifar10` , `cifar10_reduced` or`cifar100`.
* `--network` either `resnet18` , `resnet50` and `wideresnet` for WideResNet28-10

After running the required epochs, run `main_linear.py` for the learner classifier.

and example of training using Color-Spatial AutoAugment with backbone ResNet50 for 1000 epochs

```
python main.py --augment Augment --dataset cifar10 --network resnet50 --epochs 1000 --batch_size 64
```

* **Important Note:** All tests were done on batch size 64 due to memory restrictions on the testing setup.

*The table shows the accuracy of Cifar-10 with both AutoAugment and Color-Spatial AutoAugment on the different networks; for the ResNet50, we Show the accuracy for both 300 epochs and 1000.*

| **Epochs** | **Network**       | **AA** | **CS\-AA** |
|------------|-------------------|--------|------------|
|            | ResNet18          | 77\.48 | 82\.49     |
|   300      | ResNet50          | 79\.51 | 85\.74     |
|            | WideResNet 28\-10 | 79\.01 | 82\.13     |
| 	1000     | ResNet50          | 85\.56 | 91\.1      |
