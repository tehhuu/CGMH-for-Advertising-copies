# -*- coding: utf-8 -*-
class config(object):
    def __init__(self):
        self.data_path='../data/1-billion/wakati_wiki_20-50_5m.txt'            #path of data for training language model  
        self.ft_data_path = '../data/1-billion/wakati_cabocha_20-50_2.txt'
        self.use_data_path='./input/input.txt'                         #data path of keywords
        self.dict_path='../data/1-billion/dict_wakati_wiki_cabocha_20-50_5m_2.pkl'                    #dictionary path
        #self.emb_path='../data/1-billion/emb_wakati_wiki_kokoku_20-50.pkl'                  #word embedding path, used when config.sim=='word_max' or config.sim=='combine'
        self.skipthoughts_path='../skip_thoughts'                  #path of skipthoughts, used when config.sim=='skipthoughts' or config.sim=='combine'
        self.pos_path='../POS/english-models'                       #path for pos tagger
        
        self.dict_size=50000
        self.vocab_size=self.dict_size+3
        
        self.forward_save_path='./model_fte3/forward.ckpt'
        self.backward_save_path='./model_fte3/backward.ckpt'
        self.forward_log_path='./log/forward_log.txt'
        self.backward_log_path='./log/backward_log.txt'
        self.use_fp16=False
        self.shuffle=False
        self.use_log_path='./log/use_log.txt'

        self.tdict_path='../data/1-billion/dict_char_wiki_cabocha_20-50_5m_2.pkl' #単語モデルの辞書
        self.tforward_save_path='./model_c/forward.ckpt' #単語モデルの前向きモデル
        self.tbackward_save_path='./model_c/backward.ckpt' #単語モデルの後ろ向きモデル
        self.tdict_size=10762 #単語モデルの辞書サイズ
        self.tvocab_size=self.tdict_size+3

        self.batch_size=256
        self.num_steps=66
        self.hidden_size=300
        self.keep_prob=1
        self.num_layers=2
        
        self.max_epoch=10
        self.max_ft_epoch = 3 #転移学習のエポック数
        self.max_grad_norm=5
        
        self.GPU='0'
        self.mode='use'
        self.sample_time=500
        self.iter=1
        self.record_time=[100,200,300]
        self.sample_sentence_number=119
        
        self.search_size=100
        self.use_output_path='./output/output.txt'                     #output path
      
        #self.sample_prior=[1,1,1,1]
        self.action_prob=[0.3,0.3,0.3,0.1]                                         #the prior of 4 actions
        #self.threshold=0.1
        self.sim=None                                                                    #matching model
        #self.sim_word=True
        self.double_LM=False                                                           
        self.keyword_pos=False
        self.keyboard_input=False
        self.sim_hardness=5
        self.rare_since=30000
        self.just_acc_rate=0.0
        self.key_num=7
        self.min_length=20 #最低文字数
        #self.max_suggest_word=20
