import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy
import random
import math
BATCH_SIZE=256#10;
BATCH_SAMPLE=32#300
BATCH_SIZE_EVAL=256;
LR=0.003;Momentum=0.8;
FEAT=128;CATE_SIZE=1400;
#ITEM=82320;USER=1000;EMB=128;
ITEM=212487;USER=1000;EMB=128;#64+66
NEB=5;
topK=20;
SAMPLE_EVAL=1000;
rec={}
eval_set={}
test_set={}
usr_rec={}
cate_item={}
fa={};cate_item_id=[]
#"cuda:0" ##############
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainid=torch.load('/home/share/yuxuzheng/Mercari/dataset/train_id_click_men.pth').long()
def weights_init(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv3d') != -1:
		nn.init.xavier_normal_(m.weight.data)
		nn.init.constant_(m.bias.data, 0.0)
	elif classname.find('Linear') != -1:
		nn.init.xavier_normal_(m.weight.data)
		nn.init.constant_(m.bias.data, 0.0)
class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.usr=nn.Embedding(USER,128)
		self.cate=nn.Embedding(CATE_SIZE,11)
		self.cmat=nn.Embedding(CATE_SIZE,11*11)
		self.cbias=nn.Embedding(CATE_SIZE,11)
		self.ubias=nn.Embedding(USER,128)
		self.cbiasa=nn.Embedding(CATE_SIZE,11)
		self.cbiasb=nn.Embedding(CATE_SIZE,95)
		self.fc0=nn.Linear(FEAT,95,bias=True)
		self.cvismat=nn.Embedding(CATE_SIZE,95*95)
		self.cvisbias=nn.Embedding(CATE_SIZE,95)
	def forward(self,feat,fcat,fusr,fitm,flg=0):#flg=0 single user flg=1 all_user
		batlen=feat.size(0)
		cat1=self.cate(fcat[:,0].view(-1)).view(batlen,-1)
		cmat2=self.cmat(fcat[:,1].view(-1)).view(batlen,-1)
		cmat3=self.cmat(fcat[:,2].view(-1)).view(batlen,-1)
		cbias2=self.cbias(fcat[:,1].view(-1)).view(batlen,-1)
		cbias3=self.cbias(fcat[:,2].view(-1)).view(batlen,-1)
		cat2=torch.sum((cat1.repeat(1,11)*cmat2).view(batlen,11,11),2)+cbias2
		cat3=torch.sum((cat2.repeat(1,11)*cmat3).view(batlen,11,11),2)+cbias3
		cbiasa1=self.cbiasa(fcat[:,0].view(-1)).view(batlen,-1)
		cbiasa2=self.cbiasa(fcat[:,1].view(-1)).view(batlen,-1)
		cbiasa3=self.cbiasa(fcat[:,2].view(-1)).view(batlen,-1)
		cbiasb=self.cbiasb(fcat[:,2].view(-1)).view(batlen,-1)
		cbias=torch.cat([torch.zeros(batlen,33).to(device),cbiasb],1)
		cvismat=self.cvismat(fcat[:,2].view(-1)).view(batlen,-1)
		cvisbias=self.cvisbias(fcat[:,2].view(-1)).view(batlen,-1)
		itmp=torch.cat([cat1,cat2,cat3],1)
		p0=self.fc0(feat)
		itmq=torch.sum((p0.repeat([1,95])*cvismat).view(batlen,95,95),2)+cvisbias
		pitm=torch.cat([itmp,itmq],1)
		if flg==2:
			print("itmp",itmp[:2,:5])
			print("itmq",itmq[:2,:5])
		if flg==0:
			"""inc=torch.cat([self.usr(fusr.view(-1)).view(1,batlen,-1),cat3.view(1,batlen,-1),cat2.view(1,batlen,-1),cat1.view(1,batlen,-1)],0)
			ind=torch.cat([self.usr(fusr.view(-1)).view(1,batlen,-1),cat1.view(1,batlen,-1),cat2.view(1,batlen,-1),cat3.view(1,batlen,-1)],0)
			outc=self.lstmc(inc)[0].view(4,batlen,-1)
			outd=self.lstmd(ind)[0].view(4,batlen,-1)
			u1f=torch.mean(torch.cat([outc[0].view(batlen,-1,1),outd[0].view(batlen,-1,1)],2),2)
			u2f=torch.mean(torch.cat([outc[1].view(batlen,-1,1),outd[3].view(batlen,-1,1)],2),2)
			u3f=torch.mean(torch.cat([outc[2].view(batlen,-1,1),outd[2].view(batlen,-1,1)],2),2)
			u4f=torch.mean(torch.cat([outc[3].view(batlen,-1,1),outd[1].view(batlen,-1,1)],2),2)
			pusr=torch.cat([u1f,u2f,u3f,u4f],1).view(batlen,-1)"""
			tmp=torch.sum((pitm+self.ubias(fusr.view(-1)).view(batlen,-1))*(self.usr(fusr.view(-1)).view(batlen,-1)+cbias),1).view(-1,1)
			return tmp
		else:
			"""inc=torch.cat([self.usr.weight.view(1,USER,-1).repeat([batlen,1,1]).view(1,batlen*USER,-1),cat3.repeat([1,1,USER]).view(1,batlen*USER,-1),cat2.repeat([1,1,USER]).view(1,batlen*USER,-1),cat1.repeat([1,1,USER]).view(1,batlen*USER,-1)],0)
			ind=torch.cat([self.usr.weight.view(1,USER,-1).repeat([batlen,1,1]).view(1,batlen*USER,-1),cat1.repeat([1,1,USER]).view(1,batlen*USER,-1),cat2.repeat([1,1,USER]).view(1,batlen*USER,-1),cat3.repeat([1,1,USER]).view(1,batlen*USER,-1)],0)
			outc=self.lstmc(inc)[0].view(4,batlen,USER,-1)
			outd=self.lstmd(ind)[0].view(4,batlen,USER,-1)
			u1f=torch.mean(torch.cat([outc[0].view(batlen,USER,-1,1),outd[0].view(batlen,USER,-1,1)],3),3)
			u2f=torch.mean(torch.cat([outc[1].view(batlen,USER,-1,1),outd[3].view(batlen,USER,-1,1)],3),3)
			u3f=torch.mean(torch.cat([outc[2].view(batlen,USER,-1,1),outd[2].view(batlen,USER,-1,1)],3),3)
			u4f=torch.mean(torch.cat([outc[3].view(batlen,USER,-1,1),outd[1].view(batlen,USER,-1,1)],3),3)
			pusr=torch.cat([u1f,u2f,u3f,u4f],2).view(batlen,USER,-1)"""
			tmp=torch.sum((pitm.view(batlen,1,-1).repeat([1,USER,1])+self.ubias.weight.view(1,USER,-1).repeat([batlen,1,1]))*(self.usr.weight.view(1,USER,-1).repeat([batlen,1,1])+cbias.view(batlen,1,-1).repeat([1,USER,1])),2).view(batlen,USER)
			return tmp
net=Net();net.to(device)#net=net.to(device);
opt_net=torch.optim.AdamW(net.parameters(),lr=LR,betas=(0.9,0.99))#,weight_decay=0.03)
def eval():
	print("Evaling in men")
	resu=torch.Tensor(ITEM,USER)
	id=0
	while(id<ITEM):
		if id+BATCH_SIZE_EVAL<=ITEM:
			datlen=BATCH_SIZE_EVAL
		else:
			datlen=ITEM-id
		in2=torch.zeros(datlen,FEAT).float()
		in3=torch.zeros(datlen,3).long()
		in4=torch.zeros(datlen,1).long()
		for cot in range(datlen):
			in2[cot]=feat[id+cot]
			in3[cot]=cate[id+cot]
			in4[cot]=id+cot
		tmpu=net(in2.to(device),in3.to(device),"",in4.to(device),1+(id==0)).detach()
		for cot in range(datlen):
			resu[id+cot]=tmpu[cot]
		id+=BATCH_SIZE_EVAL
	lg2=[0.]
	for i in range(1,25):
		lg2.append(math.log2(i))
	print("For users:")
	#User Train
	wait_set=rec;name="Train";target=0;
	cottt=0;
	acc5=acc10=acc20=0;
	pre5=pre10=pre20=0;
	rec5=rec10=rec20=0;
	auc5=auc10=auc20=0;
	ndcg5=ndcg10=ndcg20=0;
	for i in range(USER):
		if not(i in wait_set.keys()):
			continue;
		cottt+=1;
		cot5=cot10=cot20=0;
		p5=p10=p20=0;
		q5=q10=q20=0;
		num=21
		num+=(0 if (not(i in rec.keys())) else len(rec[i]))
		num+=(0 if (not(i in eval_set.keys())) else len(eval_set[i]))
		num+=(0 if (not(i in test_set.keys())) else len(test_set[i]))
		tmpy=resu[:,i].view(-1).topk(num,largest=True)[1]
		cot=1
		for p in tmpy:
			if target!=0 and(i in rec.keys()) and (p.item() in rec[i]):
				continue
			if target!=1 and(i in eval_set.keys()) and (p.item() in eval_set[i]):
				continue
			if target!=2 and(i in test_set.keys()) and (p.item() in test_set[i]):
				continue
			if p.item() in wait_set[i]:
				if cot<=20:
					cot20+=1
					q20+=1/lg2[cot+1]
					if cot<=10:
						cot10+=1
						q10+=1/lg2[cot+1]
						if cot<=5:
							cot5+=1
							q5+=1/lg2[cot+1]
			else:
				if cot<=5:
					p5+=cot5;
				if cot<=10:
					p10+=cot10;
				if cot<=20:
					p20+=cot20;
			cot+=1
			if cot>20:
				break
		acc5+=(1 if cot5>0 else 0);acc10+=(1 if cot10>0 else 0);acc20+=(1 if cot20>0 else 0);
		pre5+=cot5/5;pre10+=cot10/10;pre20+=cot20/20;
		rec5+=cot5/len(wait_set[i]);rec10+=cot10/len(wait_set[i]);rec20+=cot20/len(wait_set[i]);
		if(cot5==5):
			auc5+=1
		elif cot5>0:
			auc5+=p5/(cot5*(5-cot5));
		if(cot10==10):
			auc10+=1
		elif cot10>0:
			auc10+=p10/(cot10*(10-cot10));
		if(cot20==20):
			auc20+=1
		elif cot20>0:
			auc20+=p20/(cot20*(20-cot20));
		p5=p10=p20=0
		for j in range(1,21):
			if j<=len(wait_set[i]):
				if j<=20:
					p20+=1/lg2[j+1]
					if j<=10:
						p10+=1/lg2[j+1]
						if j<=5:
							p5+=1/lg2[j+1]
		ndcg5+=q5/p5;ndcg10+=q10/p10;ndcg20+=q20/p20;
	f1s5 =(0 if(pre5+rec5<0.00000001) else 2*(pre5*rec5/cottt)/(pre5+rec5))
	f1s10=(0 if(pre5+rec5<0.00000001) else 2*(pre10*rec10/cottt)/(pre10+rec10))
	f1s20=(0 if(pre5+rec5<0.00000001) else 2*(pre20*rec20/cottt)/(pre20+rec20))
	print(name,"acc @5",acc5/cottt,"@10",acc10/cottt,"@20",acc20/cottt)
	print(name,"pre @5",pre5/cottt,"@10",pre10/cottt,"@20",pre20/cottt)
	print(name,"rec @5",rec5/cottt,"@10",rec10/cottt,"@20",rec20/cottt)
	print(name,"auc @5",auc5/cottt,"@10",auc10/cottt,"@20",auc20/cottt)
	print(name,"ndcg@5",ndcg5/cottt,"@10",ndcg10/cottt,"@20",ndcg20/cottt)
	print(name,"f1s @5",f1s5,f1s10,f1s20)
	#User EVAL
	wait_set=eval_set;name="EVAL ";target=1;
	cottt=0;
	acc5=acc10=acc20=0;
	pre5=pre10=pre20=0;
	rec5=rec10=rec20=0;
	auc5=auc10=auc20=0;
	ndcg5=ndcg10=ndcg20=0;
	for i in range(USER):
		if not(i in wait_set.keys()):
			continue;
		cottt+=1;
		cot5=cot10=cot20=0;
		p5=p10=p20=0;
		q5=q10=q20=0;
		num=21
		num+=(0 if (not(i in rec.keys())) else len(rec[i]))
		num+=(0 if (not(i in eval_set.keys())) else len(eval_set[i]))
		num+=(0 if (not(i in test_set.keys())) else len(test_set[i]))
		tmpy=resu[:,i].view(-1).topk(num,largest=True)[1]
		cot=1
		for p in tmpy:
			if target!=0 and(i in rec.keys()) and (p.item() in rec[i]):
				#print("pass rec ",p.item(),cate[p.item()]);
				continue
			if target!=1 and(i in eval_set.keys()) and (p.item() in eval_set[i]):
				continue
			if target!=2 and(i in test_set.keys()) and (p.item() in test_set[i]):
				#print("pass eval",p.item(),cate[p.item()]);
				continue
			#print("cot",cot,p.item(),cate[p.item()])
			if p.item() in wait_set[i]:
				if cot<=20:
					cot20+=1
					q20+=1/lg2[cot+1]
					if cot<=10:
						cot10+=1
						q10+=1/lg2[cot+1]
						if cot<=5:
							cot5+=1
							q5+=1/lg2[cot+1]
			else:
				if cot<=5:
					p5+=cot5;
				if cot<=10:
					p10+=cot10;
				if cot<=20:
					p20+=cot20;
			cot+=1
			if cot>20:
				break
		acc5+=(1 if cot5>0 else 0);acc10+=(1 if cot10>0 else 0);acc20+=(1 if cot20>0 else 0);
		pre5+=cot5/5;pre10+=cot10/10;pre20+=cot20/20;
		rec5+=cot5/len(wait_set[i]);rec10+=cot10/len(wait_set[i]);rec20+=cot20/len(wait_set[i]);
		if(cot5==5):
			auc5+=1
		elif cot5>0:
			auc5+=p5/(cot5*(5-cot5));
		if(cot10==10):
			auc10+=1
		elif cot10>0:
			auc10+=p10/(cot10*(10-cot10));
		if(cot20==20):
			auc20+=1
		elif cot20>0:
			auc20+=p20/(cot20*(20-cot20));
		p5=p10=p20=0
		for j in range(1,21):
			if j<=len(wait_set[i]):
				if j<=20:
					p20+=1/lg2[j+1]
					if j<=10:
						p10+=1/lg2[j+1]
						if j<=5:
							p5+=1/lg2[j+1]
		ndcg5+=q5/p5;ndcg10+=q10/p10;ndcg20+=q20/p20;
	f1s5 =(0 if(pre5+rec5<0.00000001) else 2*(pre5*rec5/cottt)/(pre5+rec5))
	f1s10=(0 if(pre5+rec5<0.00000001) else 2*(pre10*rec10/cottt)/(pre10+rec10))
	f1s20=(0 if(pre5+rec5<0.00000001) else 2*(pre20*rec20/cottt)/(pre20+rec20))
	print(name,"acc @5",acc5/cottt,"@10",acc10/cottt,"@20",acc20/cottt)
	print(name,"pre @5",pre5/cottt,"@10",pre10/cottt,"@20",pre20/cottt)
	print(name,"rec @5",rec5/cottt,"@10",rec10/cottt,"@20",rec20/cottt)
	print(name,"auc @5",auc5/cottt,"@10",auc10/cottt,"@20",auc20/cottt)
	print(name,"ndcg@5",ndcg5/cottt,"@10",ndcg10/cottt,"@20",ndcg20/cottt)
	print(name,"f1s @5",f1s5,f1s10,f1s20)
	#User TEST
	wait_set=test_set;name="TEST ";target=2;
	cottt=0;
	acc5=acc10=acc20=0;
	pre5=pre10=pre20=0;
	rec5=rec10=rec20=0;
	auc5=auc10=auc20=0;
	ndcg5=ndcg10=ndcg20=0;
	for i in range(USER):
		if not(i in wait_set.keys()):
			continue;
		cottt+=1;
		cot5=cot10=cot20=0;
		p5=p10=p20=0;
		q5=q10=q20=0;
		num=21
		num+=(0 if (not(i in rec.keys())) else len(rec[i]))
		num+=(0 if (not(i in eval_set.keys())) else len(eval_set[i]))
		num+=(0 if (not(i in test_set.keys())) else len(test_set[i]))
		tmpy=resu[:,i].view(-1).topk(num,largest=True)[1]
		cot=1
		for p in tmpy:
			if target!=0 and(i in rec.keys()) and (p.item() in rec[i]):
				continue
			if target!=1 and(i in eval_set.keys()) and (p.item() in eval_set[i]):
				continue
			if target!=2 and(i in test_set.keys()) and (p.item() in test_set[i]):
				continue
			if p.item() in wait_set[i]:
				if cot<=20:
					cot20+=1
					q20+=1/lg2[cot+1]
					if cot<=10:
						cot10+=1
						q10+=1/lg2[cot+1]
						if cot<=5:
							cot5+=1
							q5+=1/lg2[cot+1]
			else:
				if cot<=5:
					p5+=cot5;
				if cot<=10:
					p10+=cot10;
				if cot<=20:
					p20+=cot20;
			cot+=1
			if cot>20:
				break
		acc5+=(1 if cot5>0 else 0);acc10+=(1 if cot10>0 else 0);acc20+=(1 if cot20>0 else 0);
		pre5+=cot5/5;pre10+=cot10/10;pre20+=cot20/20;
		rec5+=cot5/len(wait_set[i]);rec10+=cot10/len(wait_set[i]);rec20+=cot20/len(wait_set[i]);
		if(cot5==5):
			auc5+=1
		elif cot5>0:
			auc5+=p5/(cot5*(5-cot5));
		if(cot10==10):
			auc10+=1
		elif cot10>0:
			auc10+=p10/(cot10*(10-cot10));
		if(cot20==20):
			auc20+=1
		elif cot20>0:
			auc20+=p20/(cot20*(20-cot20));
		p5=p10=p20=0
		for j in range(1,21):
			if j<=len(wait_set[i]):
				if j<=20:
					p20+=1/lg2[j+1]
					if j<=10:
						p10+=1/lg2[j+1]
						if j<=5:
							p5+=1/lg2[j+1]
		ndcg5+=q5/p5;ndcg10+=q10/p10;ndcg20+=q20/p20;
	f1s5 =(0 if(pre5+rec5<0.00000001) else 2*(pre5*rec5/cottt)/(pre5+rec5))
	f1s10=(0 if(pre5+rec5<0.00000001) else 2*(pre10*rec10/cottt)/(pre10+rec10))
	f1s20=(0 if(pre5+rec5<0.00000001) else 2*(pre20*rec20/cottt)/(pre20+rec20))
	print(name,"acc @5",acc5/cottt,"@10",acc10/cottt,"@20",acc20/cottt)
	print(name,"pre @5",pre5/cottt,"@10",pre10/cottt,"@20",pre20/cottt)
	print(name,"rec @5",rec5/cottt,"@10",rec10/cottt,"@20",rec20/cottt)
	print(name,"auc @5",auc5/cottt,"@10",auc10/cottt,"@20",auc20/cottt)
	print(name,"ndcg@5",ndcg5/cottt,"@10",ndcg10/cottt,"@20",ndcg20/cottt)
	print(name,"f1s @5",f1s5,f1s10,f1s20)
	"""print("For items:")
	#Item Train
	wait_set=rec;name="Train";target=0;
	cot5=cot10=cot20=0;cottt=0;
	for i in range(ITEM):
		if not(i+USER in wait_set.keys()):
			continue;
		cottt+=1;
		num=21
		num+=(0 if (not(i+USER in rec.keys())) else len(rec[i+USER]))
		num+=(0 if (not(i+USER in eval_set.keys())) else len(eval_set[i+USER]))
		num+=(0 if (not(i+USER in test_set.keys())) else len(test_set[i+USER]))
		tmpy=resu[i].view(-1).view(-1).topk(num,largest=True)[1]
		cot=1
		for p in tmpy:
			if target!=0 and(i+USER in rec.keys()) and (p.item() in rec[i+USER]):
				continue
			if target!=1 and(i+USER in eval_set.keys()) and (p.item() in eval_set[i+USER]):
				continue
			if target!=2 and(i+USER in test_set.keys()) and (p.item() in test_set[i+USER]):
				continue
			if p.item() in wait_set[i+USER]:
				if cot<=20:
					cot20+=1
					if cot<=10:
						cot10+=1
						if cot<=5:
							cot5+=1
				break
			cot+=1
			if cot>20:
				break
	print(name,"Pre5",cot5/cottt,"Pre10",cot10/cottt,"Pre20",cot20/cottt)
	#Item EVAL
	wait_set=eval_set;name="EVAL ";target=1;
	cot5=cot10=cot20=0;cottt=0;
	for i in range(ITEM):
		if not(i+USER in wait_set.keys()):
			continue;
		cottt+=1;
		num=21
		num+=(0 if (not(i+USER in rec.keys())) else len(rec[i+USER]))
		num+=(0 if (not(i+USER in eval_set.keys())) else len(eval_set[i+USER]))
		num+=(0 if (not(i+USER in test_set.keys())) else len(test_set[i+USER]))
		tmpy=resu[i].view(-1).topk(num,largest=True)[1]
		cot=1
		for p in tmpy:
			if target!=0 and(i+USER in rec.keys()) and (p.item() in rec[i+USER]):
				continue
			if target!=1 and(i+USER in eval_set.keys()) and (p.item() in eval_set[i+USER]):
				continue
			if target!=2 and(i+USER in test_set.keys()) and (p.item() in test_set[i+USER]):
				continue
			if p.item() in wait_set[i+USER]:
				if cot<=20:
					cot20+=1
					if cot<=10:
						cot10+=1
						if cot<=5:
							cot5+=1
				break
			cot+=1
			if cot>20:
				break
	print(name,"Pre5",cot5/cottt,"Pre10",cot10/cottt,"Pre20",cot20/cottt)
	#Item TEST
	wait_set=test_set;name="TEST ";target=2
	cot5=cot10=cot20=0;cottt=0;
	for i in range(ITEM):
		if not(i+USER in wait_set.keys()):
			continue;
		cottt+=1;
		num=21
		num+=(0 if (not(i+USER in rec.keys())) else len(rec[i+USER]))
		num+=(0 if (not(i+USER in eval_set.keys())) else len(eval_set[i+USER]))
		num+=(0 if (not(i+USER in test_set.keys())) else len(test_set[i+USER]))
		tmpy=resu[i].view(-1).topk(num,largest=True)[1]
		cot=1
		for p in tmpy:
			if target!=0 and(i+USER in rec.keys()) and (p.item() in rec[i+USER]):
				continue
			if target!=1 and(i+USER in eval_set.keys()) and (p.item() in eval_set[i+USER]):
				continue
			if target!=2 and(i+USER in test_set.keys()) and (p.item() in test_set[i+USER]):
				continue
			if p.item() in wait_set[i+USER]:
				if cot<=20:
					cot20+=1
					if cot<=10:
						cot10+=1
						if cot<=5:
							cot5+=1
				break
			cot+=1
			if cot>20:
				break
	print(name,"Pre5",cot5/cottt,"Pre10",cot10/cottt,"Pre20",cot20/cottt)"""
def train():
	torch_trainid=Data.TensorDataset(trainid)
	loader = Data.DataLoader(dataset=torch_trainid,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
	for epoch in range(9999):#9999
		cott1=cott2=0;cottt=0
		print("working...")
		for step,batch in enumerate(loader):
			dat=batch[0]
			datlen=len(dat)
			poslen=datlen
			tmpusr={}
			in1=torch.Tensor(datlen*4).long()
			in2=torch.Tensor(datlen*4,FEAT).float()
			in3=torch.Tensor(datlen*4,3).long()
			in4=torch.Tensor(datlen*4,1).long()
			cotnow=0
			for cot in range(datlen):
				usrid=data_click[dat[cot]][0]
				itmid=data_click[dat[cot]][1]
				in1[cotnow]=in1[cotnow+poslen]=in1[cotnow+poslen*2]=in1[cotnow+poslen*3]=usrid
				in2[cotnow]=feat[itmid]
				in3[cotnow]=cate[itmid]
				in4[cotnow]=itmid
				"""
				#neg1
				itmidq=int(random.random()*ITEM)
				in2[cotnow+poslen*4]=feat[itmidq]
				in3[cotnow+poslen*4]=cate[itmidq]
				in4[cotnow+poslen*4]=itmidq
				"""
				#neg2
				catep=fa[fa[cate[itmid][2].item()]]
				if cate_item_id[catep]==0:
					random.shuffle(cate_item[catep])
					cate_item_id[catep]=len(cate_item[catep])
				itmidq=cate_item[catep][cate_item_id[catep]-1]
				cate_item_id[catep]-=1
				in2[cotnow+poslen]=feat[itmidq]
				in3[cotnow+poslen]=cate[itmidq]
				in4[cotnow+poslen]=itmidq
				#neg3
				catep=fa[cate[itmid][2].item()]
				if cate_item_id[catep]==0:
					random.shuffle(cate_item[catep])
					cate_item_id[catep]=len(cate_item[catep])
				itmidq=cate_item[catep][cate_item_id[catep]-1]
				cate_item_id[catep]-=1
				in2[cotnow+poslen*2]=feat[itmidq]
				in3[cotnow+poslen*2]=cate[itmidq]
				in4[cotnow+poslen*2]=itmidq
				#neg4
				catep=cate[itmid][2].item()
				if cate_item_id[catep]==0:
					random.shuffle(cate_item[catep])
					cate_item_id[catep]=len(cate_item[catep])
				itmidq=cate_item[catep][cate_item_id[catep]-1]
				cate_item_id[catep]-=1
				in2[cotnow+poslen*3]=feat[itmidq]
				in3[cotnow+poslen*3]=cate[itmidq]
				in4[cotnow+poslen*3]=itmidq
				cotnow+=1
			res1=net(in2.to(device),in3.to(device),in1.to(device),in4.to(device),0)
			res2=net(in2[:poslen,:].to(device),in3[:poslen,:].to(device),"",in4[:poslen,:].to(device),1)
			lsm=nn.Softmax(dim=1)
			resp=lsm(torch.cat([res1[:poslen,:].repeat([3,1]),res1[poslen:,:]],1))
			if (step%800==0):
				print("ori_1A",res1.view(-1)[:5])
				print("ori_1B",res1.view(-1)[poslen:poslen+5])
				#print("ori_1C",res1.view(-1)[poslen*3:poslen*3+5])
				#print("ori_2 ",res2[:5,:5])
			los2=nn.CrossEntropyLoss()
			err1=torch.mean(torch.log(torch.exp((resp[:,0]-resp[:,1])*(-1))+1))
			err2=los2(res2,in1[:poslen].to(device))
			loss=1*err1#+0.3*err2
			cott1+=err1.item();cott2+=err2.item();cottt+=1
			opt_net.zero_grad()
			loss.backward()
			opt_net.step()
			#print(epoch,step,loss.to('cpu'))
		print("epoch",epoch,"Err1 MEAN=",cott1/cottt,"Err2 MEAN=",cott2/cottt)
		print("Saving...")
		#torch.save(net.state_dict(),'bak/net_new_last_men_rp.pth')
		#torch.save(opt_net.state_dict(),'bak/net_new_opt_last_men_rp.pth')
		print("Have Saved")
		if (epoch%1==0):# or (epoch>10 and epoch%2==0):			
			eval()

if __name__ == '__main__':
	print("Data Loading...")
	print("Feature Loading...")
	net.load_state_dict(torch.load('bak/net_new_last_men_rp_single.pth'))
	opt_net.load_state_dict(torch.load('bak/net_new_opt_last_men_rp_single.pth'))
	"""net.apply(weights_init)
	nn.init.xavier_normal_(net.cmat.weight.data)
	nn.init.constant_(net.cbias.weight.data,0.0)
	nn.init.xavier_normal_(net.cvismat.weight.data)
	nn.init.constant_(net.cvisbias.weight.data,0.0)"""
	feat=torch.from_numpy(numpy.loadtxt("/home/share/yuxuzheng/Mercari/dataset/visdata_by_item_men_normal_positive_pca.dat")).float().to(device)
	cate=[]
	with open("/home/share/yuxuzheng/Mercari/dataset/itemlist_wc_by_vis_men.dat",'r',encoding='utf-8')as file:
		fin=file.readlines()
		for p in fin:
			q=p[:-1].split(' ')
			cate.append(torch.tensor([int(q[1]),int(q[2]),int(q[3])]).long())
		file.close()
	data_click=[]
	evalid=torch.load('/home/share/yuxuzheng/Mercari/dataset/valid_id_click_men.pth')###EVAL
	testid=torch.load('/home/share/yuxuzheng/Mercari/dataset/test_id_click_men.pth')###TEST
	with open("/home/share/yuxuzheng/Mercari/dataset/dataset_click_men.dat",'r',encoding='utf-8')as file:
		fin=file.readlines()
		for p in fin:
			q=p[:-1].split(' ')
			data_click.append([int(q[0]),int(q[1])])
		file.close()
	for p in range(trainid.size(0)):
		p1=data_click[trainid[p]][0]
		p2=data_click[trainid[p]][1]
		if not(p1 in usr_rec):
			usr_rec[p1]=set()
		usr_rec[p1].add(p2)
	for p in range(ITEM):
		c0=cate[p][0].item();
		c1=cate[p][1].item()
		c2=cate[p][2].item()
		if not(c0 in cate_item):
			cate_item[c0]=[]
		if not(c1 in cate_item):
			cate_item[c1]=[]
		if not(c2 in cate_item):
			cate_item[c2]=[]
		cate_item[c0].append(p)
		cate_item[c1].append(p)
		cate_item[c2].append(p)
		fa[c2]=c1;fa[c1]=c0;
	for p in range(CATE_SIZE):
		cate_item_id.append(0)
	for p in range(trainid.size(0)):
		p1=data_click[trainid[p]][0]
		p2=data_click[trainid[p]][1]
		if p1 in rec.keys():
			rec[p1].add(p2)
		else:
			rec[p1]=set([p2])
		if p2+USER in rec.keys():
			rec[p2+USER].add(p1)
		else:
			rec[p2+USER]=set([p1])
	cot=0
	for p in range(evalid.size(0)):
		p1=data_click[evalid[p]][0]
		p2=data_click[evalid[p]][1]
		if p1 in eval_set.keys():
			eval_set[p1].add(p2)
		else:
			eval_set[p1]=set([p2])
		if p2+USER in eval_set.keys():
			eval_set[p2+USER].add(p1)
		else:
			eval_set[p2+USER]=set([p1])
	for p in range(testid.size(0)):
		p1=data_click[testid[p]][0]
		p2=data_click[testid[p]][1]
		if p1 in test_set.keys():
			test_set[p1].add(p2)
		else:
			test_set[p1]=set([p2])
		if p2+USER in test_set.keys():
			test_set[p2+USER].add(p1)
		else:
			test_set[p2+USER]=set([p1])
	print("Loading Finished")
	print("Start training")
	eval()
	train()
	print("Finish training")
#torch.save(net1.fc1.state_dict(),'bak/fc1.pth')
#torch.save(net1.fc2.state_dict(),'bak/fc2.pth')
#torch.save(net1.he.state_dict(),'bak/he.pth')
