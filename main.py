import numpy as np
!pip install hmmlearn
from hmmlearn import hmm
from collections import Counter

russian_alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
start_probs = np.array((0.081, 0.0159, 0.0454, 0.017, 0.0298, 0.0794, 0.004, 0.0094, 0.0165, 0.0735, 0.0121, 0.0349, 0.04, 0.0321, 0.067, 0.1097, 0.0281, 0.0473, 0.0547, 0.0626, 0.0262, 0.0026, 0.0097, 0.0048, 0.0144, 0.0073, 0.0036, 0.004, 0.019, 0.0174, 0.0032, 0.0064, 0.021))

# Also assume it's more likely to stay in a state than transition to the other
def calculate_conditional_probabilities(text):
    text = text.lower()
    n = len(text)
    letter_count = Counter(text)

    conditional_probabilities = {}
    for letter in letter_count:
        if (letter not in russian_alphabet):
            continue
        conditional_probabilities[letter] = {}
        for second_letter in letter_count:
            if (second_letter not in russian_alphabet):
                continue
            conditional_probabilities[letter][second_letter] = (text.count(letter+second_letter) / letter_count[letter]) if letter_count[letter] != 0 else 0
    return conditional_probabilities

text = open("war_and_peace.ru(1).txt", "r").read()
conditional_probs = calculate_conditional_probabilities(text)

arr = [[0 for i in range(len(russian_alphabet))] for j in range(len(russian_alphabet))]
for letter, probs in conditional_probs.items():
    for second_letter, prob in probs.items():
        arr[russian_alphabet.index(letter)][russian_alphabet.index(second_letter)] = prob
transprob = np.array(arr)
trans_mat = transprob / np.sum(transprob, axis=1).reshape(-1, 1)

model = hmm.CategoricalHMM(n_components = 33, init_params='e', params = 't', tol=0.00000000000000000001, algorithm = "map")
model.transmat_ = trans_mat
model.startprob_ = start_probs

# ТВОЙ ШИФР БЛЯТСКИЙ
str = "МКМЭФПФЯБУГМЫРЪКЪЮСЁРКЬЩЬУТЪЛБЮЬГМЫЪЩФПОГЮЪПОЮФЕБЖЭФЩМЕЪЮСЁДАЦНФГОЩЛФТЪЛОЭЯМХФПОГЪЩЮЬЛЯЬДФЛЪАЮБРЬДФГОЩЮЬКЪДФАЭЬРЭМХФДФЯОЩЩЬПЕМГЪАЬЩЬПКЪВОЁЫЯФГОЩЯМАЬКЭЬЮЬДЪЩДУНАОГОЩЩЬПЮЬЮЪДЪГФПСНГФЯФЗЭЬНАКЪГСЮЪДОГЬЮЮСНРДЪЯЪЁОРЫМВЭЬЩЬПЮЬЭМЯБОНЮФЗЭЬНАЩФОЩЫЪРФЭФЮЫЪРГДЪЯЪЁЩЬПКЪАОГФКДОГЪЮОЁЛФКЮСЩЬПФРЬЯЪЛЯОНКСЮМЩДФКЮСЮЬЫЯЪХЛЪАЕЬЮСЁОЛМАЩФЁОЩЯОГТЬЩБДОЩУРЪЁЛЯЪЭЯЬАЮСНЕЯЪГФЁОРДФГДСНФГУЩУАЮСНОАЮОПОГУГБЭЬОНПФЯАЭФЁЩЬПЭФЯФКЪДОЕПОПФНФГФПЛКЪЮУЪЩХЯФРЮФХФТЬЯУЩЬПДФЫКЬЭЬНЛЪЯЪГЮЬЯФГФПЕЪЯЪРКЪАЬЕЪЯЪРПФЯУЭФКГМЮЮЪАЪЩЫФХЬЩСЯУДЩЪПЮОТЪЩЬПТЬЯЪДЮЬЩМЗОЩЬЫМЯСЁДФКЭЪЁДЪЯЮФАКМЗОЩЩЬПАЩМЛЬАЫЬЫФЖУХФЁОГЪЩЫЯЪГЪЩАЬПЬАФЫФЁЩЬПТЬЯБЭЬШЪЁЮЬГРКЬЩФПЕЬНЮЪЩЩЬПЯМААЭФЁГМНЩЬПЯМАБЖЛЬНЮЪЩОЩЬПУЫСКОПЪГУЛОКМПФЯУДОГЪКГМЫРЪКЪЮСЁЛФГЮОПАОГЪКОЭФЩМЕЪЮСЁАДФОПЮЪАЭЬРЭОХФДФЯОКФГЮМУЛФПЮЖАЭЬРЭМЧЩМЛФДЪГЬЖЩЪЛЪЯБУАДЪЩМГЪКЬГЬДЮФПОЮМДВОНГЮЪЁЛЯЪГЬЮБУАЩЬЯОЮСХКМЫФЭФЁДЩФКЛЪПФХМЕОНАСЮФДЪЁАГЯМРБУПОДХЯОГЮОТЪДСАФЭФЁДКЬГОПОЯАФКЮТЪЛОЯФДЬКПЪЮБВМЖГФЕБФЮДСГЬДЬКРЬЭЮУРУНЯЬЫЯФХФЯМАКЬЮЬОПЪГОРЩУЗЭФХФАЩЬЭЬЮЬРЬОНРГФЯФДБЪДСЛОДЬКЮЪАЭФЯФЪКОЛЯЪГЭОЮЬВОЮЪАЭФЯФГДОХЬКОАБЭЯМХФПЭФДВОАЪЯЪЫЯУЮСЪЕЬВОАЭОЛУШОПЛОДФПОДОЮФПФЮОДЪАЪКБЪДАЪЯГТЪКОКОВОЛЪКЬЛЪЮЬЛФЭЯЬУПОНДЬЗЮФЕЬВЮОЭОЮФАОКООЮОРЭФЭКЬЮУКОАБХФАЩУПАКОКОАУЯЪЕОДВМПЮЪДЮУЩЮСЁЗМЗЗОЩХФАЩЪЁДЪАЪКСЁЭЯМХЮФДГЯМХЯЬРГЬКАУХКЬАЛЯОУЩЮСЁОРДФЮЭОНХМАКЪЁЫЪХКСЁРДМЭДАЪАПФКЭКОАКМВЬЖЩЫЬУЮЬОАКЬДОЩАКЬГФАЩЮСЁЛЪДЪТКЖГПОКМЛЯЪКЪАЩБОЯМАКЬЮЬОКЪКЪПАДОЩСЁОПДЪЮЪТЮФАЩЯЬАЩБЖЛСКЭФЁМЩФПКЪЮЮСЁЮЪЪАЩЮЪЛБЪЩЯМАКЬЮДКЖЫКЪЮЮСЁЮЬГЯМХЬПОКФХФХКУГОЩДРГСНЬЪЩАЪЯГОЩАУХФЯОЩОШОЛКУМАФЩЮЪЩЪЯЛЪЮБУАЕОЩЬЪЩЭЬЗГСЪПХЮФДЪЮБУДМЮСЮБОАЛЬАПМЯЮСПЕЪКФПРЬВМПЮСПАДЬГЪЫЮСПАЩФКФПАОГУЩЩЯОДОЩУРУПКЬГСЪЫЪРПФКДЮСРЬЭФДВФПЛМАЩСПРЬЫСКОЭМЫЭОЭЯМХФДСЪОЫЯЬВЮЬЮЪЛЯОУЩЮСОПЮЪАКСВЬЩДЪШЪХФЫЬУЮЬЛФЩМЛОКОАПМШЪЮЮСЁДРХКУГЩФЩЯОАФЛЪЯЮОЭЬЯМАКЬЮЬДГМВЪЮЪАЕЬАЩЮСЪЩЬУЩКЖЫДООЮЪЮЬДОАЩОУГФГОЮЯФХГЬЁДФОЩЪКБАПЪКСЁПЪЕФПЯЬРГДОЮМДВОЁЛЯЪГЪКСЫФХЬЩСНЭОЪДАЭОНЛФКЪЁГЯМХФЁЙЬЯКЬЙЭЯОЭМЮЮЬГПЪЮЮСЁДЛОЯЬНЮОЭЪПЮЪЛФЫЪЗГЪЮЮСЁЮФДФОЮАЭЯФПЮСЁАЯЪГБПЪЕЪЁЛФАКЪГЮОЁЛФКЮСЁАЩЯЬАЩЮФЁГМПСПКЬГФЁНЬРЬЯАЭОЁНЬЮЯЬЩПОЯДАЪЩЯФЪЫКЪГЮСОМХЯЖПСОЛОЯДЪАЪКСЁОПЮЪДЛОЯ"
observation_sequence = np.array([russian_alphabet.index(i.lower()) for i in str]).reshape(-1, 1)

X = []
row = []
cnt = 0
for i in observation_sequence:
  if cnt != 33:
    row.append(i)
  else:
    X.append(row)
    row = []
    cnt = 0
  cnt +=1

Z = X[0]
for i in range(1, 56):
  Z = np.concatenate([X[0], Z])
# data = np.array(X[0], dtype=int)

# # pretend this is repeated, so we have more data to learn from:
# lengths = [len(X)] * 5
# sequences = np.tile(data, (5,1))

model.fit(Z)

logprog, received = model.decode(observation_sequence, len(observation_sequence))
print("MOST FUCK : ", "".join([russian_alphabet[i] for i in received]))
