from Fiber import Fiber
from Transmitter import Tx
import config
tx = Tx()
fb = Fiber(tx.lam_set,length=1e5,alphaB=0.2,n2=2.7e-20,disp=17,dz=100,Nch=config.Nch,generate_noise=True)