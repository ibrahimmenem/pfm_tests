
class kp:
    x=1
    y=2
    def __init__(self,x,y):
         self.x=x
         self.y=y

if __name__ == '__main__':
	
    a1=kp(8,2)
    a2=kp(3,4)
    a3=kp(3,6)
    a4=kp(4,3)
    d=[11,22,33,33]
    l1= [a1, a2, a3 ,a4]
    ziped=zip(l1,d)
    for i in range(4): 
           print l1[i].x, l1[i].y
    print "______________"
    print ziped
    l2=sorted(ziped, key=lambda z: z[0].x, reverse=True)
    for i in range(4): 
           print l2[i].x, l2[i].y