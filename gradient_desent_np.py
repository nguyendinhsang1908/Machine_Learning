import numpy as np

# X^2+2x+4
def loss(theta):
  return theta**2+2*theta+4

def DaohamLoss(theta):
  return 2*theta+2

theta=-5
alpha=0.0001
espo=0.0000001

while True:
  theta=theta-alpha*DaohamLoss(theta)
  if abs(DaohamLoss(theta))<espo:
    break
  
print(theta)
print(loss(theta))