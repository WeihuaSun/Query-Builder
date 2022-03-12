# Query-Builder
����csv��ʽ�������ݣ���������"=,<=,>=,[]"��sql��ѯ��

## ���ɲ�ѯ����

1.�ӱ���ѡ��ĳЩ���ԣ�


�����ṩ�����ַ���ѡ��ĳЩ���ԣ�
|����|ʵ�ַ�ʽ|��ѡ����|
|---|---|---|
|asf_pred_number|�ڱ��������ѡ��n����ͬ�����ԡ�����ͨ���������(nums)�������ѡ��������Ҳ����ͨ���������Ժ�����(blacklist)�������(whitelist)������ѡ����|whitelist�����԰�����      blacklist:���Ժ�����     nums���������ѡ������|
|asf_comb|ѡ��ʹ�������������(comb)|comb:����������б�|

2.Ϊ��Щ����ѡ�����ĵ㣻

|����|ʵ�ַ�ʽ|��ѡ����|
|---|---|---|
|csf_distribution|�����ڱ������ѡ��1000����Ϊ���ĵ�ĺ�ѡ�С���������Ҫ���ĵ�ʱ��ѡ��ĳһ�ж�Ӧ���Ե�ֵ��Ϊ�����ĵ㡣|data_from������һ�п�ʼѡ��|
|csf_vocab_ood|Ϊÿ�����Ե������ѡ��һ������Ϊ���ĵ�|-|

3.�������Ժ����ĵ�ѡ��һ����ȡ�

|����|ʵ�ַ�ʽ|��ѡ����|
|---|---|---|
|wsf_uniform|�������е����ֵ����Сֵ֮����ʾ��ȵ�ѡ��һ��ֵ��Ϊ���width,Ȼ����ݿ��ѡ��<=,>=,=,[]��Ϊ��Ӧν�ʡ�Nan��Nat,�Լ�null��ֻ��ѡ��"="��Ϊν�ʡ�|-|
|wsf_exponential|ͨ��ָ���ֲ�ѡ����ֵ|-|
|wsf_naru|1.���ȴ�"="">=""<="֮�����ѡ��һ��������2.ѡ���Ӧ���Ե����ĵ���Ϊ��������������ֵ������������10�����ѡ�����ν�ʡ�|-|
|wsf_equal|ֻѡ�����ν��|-|


## �ļ�˵��

|�ļ�|����|
|---|---|
|constants.py|����Ŀ¼��صĳ���|
|dataset.py|���ݼ���أ������Ա��еĽ���|
|dtype.py|���������͵Ľ���|
|query.py|��ѯ�ı�ʾ|
|generator.py|�ڱ������ɲ�ѯ|
|main.py|����sql��ѯ��|


```mermaid
graph TD
A[����] -->B(����ѡ��)
    B --> asf_pred_number
    B --> asf_comb
  	asf_pred_number -->C[���ĵ�ѡ��]
    asf_comb -->C[���ĵ�ѡ��]
    C --> csf_distribution
    C --> csf_vocab_ood
    csf_distribution --> D[���ѡ��]
    csf_vocab_ood --> D[���ѡ��]
    D --> wsf_uniform 
    D --> wsf_exp
    D -->wsf_naru
    D -->wsf_equal
    wsf_uniform -->E[��ѯ��]
    wsf_exp-->E[��ѯ��]
    wsf_naru-->E[��ѯ��]
    wsf_equal-->E[��ѯ��]
    E-->SQL��ѯ
   
```



## ���ʹ��

dataĿ¼�´��Ҫ��ȡ�ĵ������ݣ����������outputĿ¼�¡�
```bash
python -m main [--s <seed>] [--d <dataset>]  [-q <query>] [--params <params>]
"""
Options:
  --s <seed>                Random seed.
  --d <dataset>             The input dataset [default: census].
  --q <query>               Name of the sqls [default: newquery].
  --n <number>              The number of queries to be generated[default: 10000].
  --params <params>         Parameters that are needed.
"""
```
���У�params�Ƕ���������ĵ�ѡ����������ʽ���£�
```bash
params:" {  'attr': {'pred_number': p1[, 'fun2': p2]}, \
            'center': {'distribution': p1[, 'fun2': p2]}, \
            'width': {'uniform': p1[, 'fun2': p2]},\
            'attr_params': {['param':]},\
            'center_params':{['param':]}\
            'width_params':{['param':]}\
            } "
```
����'attr''center''width'�ֱ�ʱ���ɲ�ѯ���������裬����Dict�������{���ɷ�ʽ�����ø÷�ʽ�ĸ���}�����磬ѡ��pred_number��1.0�ĸ���ѡ�����ԣ����Խ���������Ϊ�� 'attr': {'pred_number': 1.0}��
'attr_params''center_params''width_params'�����������ض��Ĳ�������Ӧ���������ݡ�

- ���磬��cencus���ݼ�����10000����ѯ��������ѯ�ļ�����Ϊqueries.sql,����ִ����������
```bash
python -m main --d census --q queries --n 10000 
```
- Ҳ���Զ����ɷ�ʽ����ѡ������Ҫ��������paramһЩ���ݡ����磬�����census���'age','workclass'������ѡ�����ԣ�����ʹ��distribution��vocab_ood���������������ģ������������
```bash
  python -m main --params "{'attr': {'pred_number': 1.0}, \
                        'center': {'distribution': 0.9, 'vocab_ood': 0.1}, \
                        'width': {'uniform': 0.5, 'exponential': 0.5}\
                        'attr_params':{'whitelist':['age','workclass']},\
                        }"
```
