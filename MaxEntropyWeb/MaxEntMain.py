from MaxEntropyWeb import MaxEnt

__author__ = 'wangjiewen'

model = MaxEnt.MaxEnt()

model.load_data('train.txt')
model.train()

print
print '---------------------'

probA = model.predict('Sunny')
print probA

probB = model.predict('Rainy')
print probB