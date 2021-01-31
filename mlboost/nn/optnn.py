#!/usr/bin/python
from . import nn
from numpy import *

_lr = nn._lr
_opt = nn._opt
from .nn import getsortedidx

class OptNN(nn.NeuralNet):

    def bprop(self, target, lr=_lr, name='std', opt=_opt, mode=1):
        ''' target is a class idx or the value in regression '''
        self.init_bprop(target)
        
        
        if name == 'std':
            self.bprop_std(self.targets, lr, opt)
        elif name == 'fraka1':
            self.bprop_fraka1(self.targets, lr, opt, mode)
        elif name == 'fraka2':
            self.bprop_fraka2(self.targets, lr, opt, mode)
        elif name == 'fraka3':
            self.bprop_fraka3(self.targets, lr, opt, mode)
        elif name == 'fraka4':
            self.bprop_fraka4(self.targets, lr, opt, mode)
        elif name == 'fraka5':
            self.bprop_fraka5(self.targets, lr, opt, mode)
        elif name == 'jeremy':
            self.bprop_jeremy(self.targets, lr, opt, mode)
        else:
            raise "unknown opt name %s"
        
        return self.cost(self.targets)

    def bprop_fraka1(self, targets, lr=_lr, opt=_opt, mode = 1):
        ''' '''

        self.compute_deltas(targets)
        output_idx = getsortedidx(self.output_deltas)
        hidden_idx = getsortedidx(self.hidden_deltas[:-1])
        
        # update output weights
        for o in output_idx:
             # update output bias
            self.bo[o] += lr * self.output_deltas[o] 
            for h in range(self.nh-1):
                self.compute_output_deltas(targets)
                self.bh[h] += lr * self.hidden_deltas[h]  
                delta = lr * self.output_deltas[o]*self.ah[h]
                self.wo[h,o] += delta
                # fprop hidden->outputs
                if mode == 1:
                    self.ao = self.outfct(dot(self.ah, self.wo))
                else:
                    self.hofprop(h, o, delta)
                
        
        output_idx = getsortedidx(self.output_deltas)
        hidden_idx = getsortedidx(self.hidden_deltas[:-1])

        # update input weights
        for o in output_idx:
            self.compute_deltas(targets, o)
            for h in hidden_idx:
                for i in range(self.ni):
                    self.compute_deltas(targets, o)
                    delta = lr * self.hidden_deltas[h]*self.ai[i]
                    self.wi[i,h] += delta
                    if mode == 1:
                        self.fprop(self.inputs, opt) 
                    else:
                        self.ihfprop(i, h, delta)

 
    def bprop_fraka2(self, targets, lr=_lr, opt=_opt, mode=1):

        self.compute_deltas(targets)
        output_idx = getsortedidx(self.output_deltas)

        # update output weights
        for o in output_idx:
            # update output bias
            self.bo[o] += lr * self.output_deltas[o]    
            hidden_idx = getsortedidx(self.hidden_deltas[:-1])
            for h in hidden_idx:
                self.compute_deltas(targets, o)
                self.bh[h] += lr * self.hidden_deltas[h]    
                self.wo[h,o] += lr * self.output_deltas[o]*self.ah[h]
                
                # fprop hidden->outputs
                self.ao = self.outfct(dot(self.ah, self.wo))
                # update input weights
                for i in range(self.ni):
                    self.compute_deltas(targets, o)
                    self.wi[i,h] += lr * self.hidden_deltas[h]*self.ai[i]
                    self.fprop(self.inputs, opt)

    def bprop_fraka3(self, targets, lr=_lr, opt=_opt, mode=1):

        self.compute_deltas(targets)
        output_idx = getsortedidx(self.output_deltas)

        # update output weights
        for o in output_idx:
            self.compute_deltas(targets, o)
            # update output bias
            self.bo[o] += lr * self.output_deltas[o]    
            hidden_idx = getsortedidx(self.hidden_deltas[:-1])
            for h in hidden_idx:
                # update hidden bias 1 time
                if o==output_idx[0]:
                    self.bh[h] += lr * self.hidden_deltas[h]    
                self.wo[h,o] += lr * self.output_deltas[o]*self.ah[h]
                
                # update input weights
                self.wi[:,h] += lr * self.hidden_deltas[h]*self.ai
                self.fprop(self.inputs, opt)

    def bprop_fraka4(self, targets, lr=_lr, opt=_opt, mode=1):
        ''' 1 output at a time '''
        self.compute_deltas(targets)
        output_idx = getsortedidx(self.output_deltas)

        # only good class and error class
        if mode == 3:
            ridx = self.result()
            if ridx != self.target:
                output_idx = [self.target, ridx]
            else:
                return

        # update output weights
        for o in output_idx:
            self.compute_deltas(targets, o)
            if mode == 1:
                self.wo += lr * dot(matrix(self.ah).T, 
                                    matrix(self.output_deltas))
            else:
                self.bo[o] += lr * self.output_deltas[o]
                self.wo[:,o] += lr * self.output_deltas[o]*self.ah
                
            self.wi += lr * dot(matrix(self.ai).T, 
                                matrix(self.hidden_deltas))
            self.fprop(self.inputs, opt) 

    def bprop_fraka5(self, targets, lr=_lr, opt=_opt, mode=1):
        ''' 1 hidden at a time '''
        self.compute_deltas(targets)
        hidden_idx = getsortedidx(self.hidden_deltas)

        # update output weights
        for h in hidden_idx:
            self.compute_deltas(targets)
            self.wo[h,:] += lr * self.output_deltas*self.ah[h]
            self.wi[:,h] += lr * self.hidden_deltas[h]*self.ai
            self.fprop(self.inputs, opt)     
    
    def bprop_jeremy(self, targets, lr=_lr, opt=_opt, mode=2):

        self.compute_deltas(targets)
        output_idx = getsortedidx(self.output_deltas)
        hidden_idx = getsortedidx(self.hidden_deltas)

        # update output weights
        for k in output_idx:
            for j in range(self.nh):
                self.compute_output_deltas(targets)
                self.wo[j,k] += lr * self.output_deltas[k]*self.ah[j]
                # fprop hidden->outputs
                self.ao = self.outfct(dot(self.ah, self.wo))
        
        output_idx = getsortedidx(self.output_deltas)
        hidden_idx = getsortedidx(self.hidden_deltas)

        if mode == 3:
            self.compute_deltas(targets)

        # update input weights
        for k in output_idx:

            if mode == 2:
                self.compute_deltas(targets, k)

            for j in hidden_idx:
                for i in range(self.ni):

                    if mode == 1:
                        self.compute_deltas(targets, k)

                    self.wi[i,j] += lr * self.hidden_deltas[j]*self.ai[i]

                    if mode == 1:
                        self.fprop(self.inputs, opt) 

            if mode == 2:
                self.fprop(self.inputs, opt) 

        if mode == 3:
            self.fprop(self.inputs, opt)


if __name__ == '__main__':
    nn.NeuralNet = OptNN
    nn.main()
