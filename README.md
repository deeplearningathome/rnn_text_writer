## About
See [full blog post here.](http://deeplearningathome.com/2016/10/Text-generation-using-deep-recurrent-neural-networks.html) 
You will need to have Tensorflow installed and, preferably, have a high-end NVIDIA GPU in case you'd like to do your own training.
## Example Run
**TRAINING** (char model on tiny text)
```python
python ~/repos/rnn_text_writer/ptb_word_lm.py --data_path=DSpeeches --file_prefix=dtrump --seed_for_sample="make america" --model=charlarge --save_path=charlarge
```
**SAMPLING MODE** (this assumes that there is trained model in "save_path"
```python
python ~/repos/rnn_text_writer/ptb_word_lm.py --data_path=DSpeeches --file_prefix=dtrump --seed_for_sample="make america" --model=charlarge --save_path=charlarge --sample_mode=True
```