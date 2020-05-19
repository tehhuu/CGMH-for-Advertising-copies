from collections import Counter
import tensorflow as tf
import pickle as pkl

tf.flags.DEFINE_string('mode','make','make or use or reverse')
tf.flags.DEFINE_string('src_path','wakati_cabocha_20-50_2_wp.txt','')
tf.flags.DEFINE_string('src_org_path','wakati_mainichi_20-50.txt','')
tf.flags.DEFINE_string('tgt_path','output','')
tf.flags.DEFINE_string('dict_path','dict_01.pkl','')
tf.flags.DEFINE_integer('dict_size',50000,'')
FLAGS=tf.flags.FLAGS

def main(_):
    f=open(FLAGS.src_path)
    f_org=open(FLAGS.src_org_path)
    if FLAGS.mode=='make':
        words=f.read().split()
        counter=Counter(words)
        common_words=counter.most_common()
        print len(common_words)       
        common_words=zip(*common_words)[0]
        word2id=dict(zip(common_words, range(len(common_words))))
        id2word=dict(zip(range(len(common_words)), common_words))

        words=f_org.read().split()
        counter=Counter(words)
        org_common_words=counter.most_common()
        print len(org_common_words)
        org_common_words=zip(*org_common_words)[0]
        common_words_ex=[word for word in org_common_words if word not in common_words]
        print(len(common_words_ex))
        org_common_words=common_words_ex[:FLAGS.dict_size-len(common_words)]
        org_word2id=dict(zip(org_common_words, range(len(common_words), FLAGS.dict_size)))
        org_id2word=dict(zip(range(len(common_words), FLAGS.dict_size), org_common_words))
        print(len(org_word2id),len(org_id2word))

        word2id.update(org_word2id)
        id2word.update(org_id2word)
        print(len(word2id),len(id2word))

        with open(FLAGS.dict_path,'w') as f:
            pkl.dump([word2id, id2word] ,f)
        Bo=[(x in word2id) for x in words]
        print sum(Bo)/(len(Bo)+0.0) 
        
    if FLAGS.mode=='use':
        Dict, _=pkl.load(open(FLAGS.dict_path))
        dict_size=len(Dict)
        with open(FLAGS.tgt_path,'w') as g:
            output_list=[]
            for line in f:
                word_list=line.split()
                for i in range(len(word_list)):
                    word=word_list[i]
                    if word in Dict:
                        word_list[i]=Dict[word]
                    else:
                        word_list[i]=dict_size
                output_list.append(word_list)
            if FLAGS.tgt_path[-3:]=='pkl':
                pkl.dump(output_list, g)
            else:
                for line in output_list:
                    for i in range(len(line)):
                        g.write(str(line[i]))
                        if i<len(line)-1:
                            g.write(' ')
                        else:
                            g.write('\n')
    
    if FLAGS.mode=='reverse':
        _, Dict=pkl.load(open(FLAGS.dict_path))
        dict_size=len(Dict)
        with open(FLAGS.tgt_path,'w') as g:
            output_list=[]
            for line in f:
                word_list=line.split()
                for i in range(len(word_list)):
                    word=int(word_list[i])
                    if word in Dict:
                        word_list[i]=Dict[word]
                    else:
                        word_list[i]='UNK'
                output_list.append(word_list)
            if FLAGS.tgt_path[-3:]=='pkl':
                pkl.dump(output_list, g)
            else:
                for line in output_list:
                    for i in range(len(line)):
                        g.write(line[i])
                        if i<len(line)-1:
                            g.write(' ')
                        else:
                            g.write('\n')

if __name__=='__main__':
    tf.app.run()

