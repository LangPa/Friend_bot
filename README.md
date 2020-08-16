## Character-level RNN with Facebook data

Create a bot to imitate and continue prose in the style of your facebook messenger chats using a [character-level LSTM RNN](https://en.wikipedia.org/wiki/Long_short-term_memory). Built with Python PyTorch package.

A demonstration of the training process may be found in the Train_model.ipynb notebook

## Example:


```
>>>print(predict_model.sample(net, 1000, prime='lang', top_k=5))

Love u did in that then then
Lol them stull is a forget an the curd
Lol
Indenting sandig lang
Yh hear u went it
I do it as my room
I did its a thower in
What i weer
Where u said
Lol they call starts
Wat mate mate
They warking torry
I have a few mines of there in library
In lol
Sak
I have to go stop the peng
With up to go they they doing them
Sure me
I shudve a bo leave it
Was i doing to start to come but its a they day
Walk same to hand the recker i week in a funded
Well will see work sem up on so tord it
Then suco u
It doing
Shope is it to go shade
```

## Installation and Setup Instructions

Clone down this repository. You will need Python and `Pytoch` installed on your machine.  To install Pytorch follow the instructions found [here](https://pytorch.org/), it is recomended that you use a CUDA enabled device to run the training of the model.

To access your facebook data do the following:
 From the FaceBook website go to Settings -> Your FaceBook Information -> Download your information. From here you may tailor how much data you would like to download. Note typically these files are very large in size and that FaceBook may take a few days to give you the data.

For this project all that is required is your messages data, downloaded as a high quality JSON file.

