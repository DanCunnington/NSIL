# Setup script to download datasets
mkdir data && cd data
curl -O -L https://www.dropbox.com/s/6yjucprsvcu6se4/fashion_mnist_1_to_5.zip
curl -O -L https://www.dropbox.com/s/lj62jxl3vqeh9kn/mnist_digits_1_to_5.zip
curl -O -L https://www.dropbox.com/s/xhadba859spic3f/MNIST.zip
unzip -q fashion_mnist_1_to_5.zip && rm fashion_mnist_1_to_5.zip
unzip -q mnist_digits_1_to_5.zip && rm mnist_digits_1_to_5.zip
unzip -q MNIST.zip && rm MNIST.zip

# Download Meta_Abd data
git clone https://github.com/AbductiveLearning/Meta_Abd.git
cp -r Meta_Abd/data/monadic ../examples/recursive_arithmetic/data/
find ../examples/recursive_arithmetic/data/ -name *.yaml -type f -exec sed -i -e 's/data/Meta_Abd_data/g' {} \;
rm -fr Meta_Abd
cd ../