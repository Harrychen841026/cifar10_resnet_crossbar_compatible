�� 
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02v2.1.0-0-ge5bf8de8��
�
activation_quant_15/reluxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameactivation_quant_15/relux

-activation_quant_15/relux/Read/ReadVariableOpReadVariableOpactivation_quant_15/relux*
_output_shapes
: *
dtype0
�
conv2d_noise_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameconv2d_noise_18/kernel
�
*conv2d_noise_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_noise_18/kernel*&
_output_shapes
:@@*
dtype0
�
conv2d_noise_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameconv2d_noise_18/bias
y
(conv2d_noise_18/bias/Read/ReadVariableOpReadVariableOpconv2d_noise_18/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_16/gamma
�
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_16/beta
�
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_16/moving_mean
�
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_16/moving_variance
�
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes
:@*
dtype0
�
activation_quant_16/reluxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameactivation_quant_16/relux

-activation_quant_16/relux/Read/ReadVariableOpReadVariableOpactivation_quant_16/relux*
_output_shapes
: *
dtype0
�
conv2d_noise_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameconv2d_noise_19/kernel
�
*conv2d_noise_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_noise_19/kernel*&
_output_shapes
:@@*
dtype0
�
conv2d_noise_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameconv2d_noise_19/bias
y
(conv2d_noise_19/bias/Read/ReadVariableOpReadVariableOpconv2d_noise_19/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_17/gamma
�
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_17/beta
�
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_17/moving_mean
�
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_17/moving_variance
�
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes
:@*
dtype0
�
activation_quant_17/reluxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameactivation_quant_17/relux

-activation_quant_17/relux/Read/ReadVariableOpReadVariableOpactivation_quant_17/relux*
_output_shapes
: *
dtype0
�
conv2d_noise_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameconv2d_noise_20/kernel
�
*conv2d_noise_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_noise_20/kernel*&
_output_shapes
:@@*
dtype0
�
conv2d_noise_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameconv2d_noise_20/bias
y
(conv2d_noise_20/bias/Read/ReadVariableOpReadVariableOpconv2d_noise_20/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_18/gamma
�
0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_18/beta
�
/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes
:@*
dtype0
�
"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_18/moving_mean
�
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes
:@*
dtype0
�
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_18/moving_variance
�
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes
:@*
dtype0
�
activation_quant_18/reluxVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameactivation_quant_18/relux

-activation_quant_18/relux/Read/ReadVariableOpReadVariableOpactivation_quant_18/relux*
_output_shapes
: *
dtype0
�
dense_noise/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*#
shared_namedense_noise/kernel
y
&dense_noise/kernel/Read/ReadVariableOpReadVariableOpdense_noise/kernel*
_output_shapes

:@
*
dtype0
x
dense_noise/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_noise/bias
q
$dense_noise/bias/Read/ReadVariableOpReadVariableOpdense_noise/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
 Adam/activation_quant_15/relux/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_15/relux/m
�
4Adam/activation_quant_15/relux/m/Read/ReadVariableOpReadVariableOp Adam/activation_quant_15/relux/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_18/kernel/m
�
1Adam/conv2d_noise_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_18/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_18/bias/m
�
/Adam/conv2d_noise_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_18/bias/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_16/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_16/gamma/m
�
7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_16/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_16/beta/m
�
6Adam/batch_normalization_16/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/m*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_16/relux/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_16/relux/m
�
4Adam/activation_quant_16/relux/m/Read/ReadVariableOpReadVariableOp Adam/activation_quant_16/relux/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_19/kernel/m
�
1Adam/conv2d_noise_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_19/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_19/bias/m
�
/Adam/conv2d_noise_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_19/bias/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_17/gamma/m
�
7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_17/beta/m
�
6Adam/batch_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/m*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_17/relux/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_17/relux/m
�
4Adam/activation_quant_17/relux/m/Read/ReadVariableOpReadVariableOp Adam/activation_quant_17/relux/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_20/kernel/m
�
1Adam/conv2d_noise_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_20/kernel/m*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_20/bias/m
�
/Adam/conv2d_noise_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_20/bias/m*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_18/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_18/gamma/m
�
7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/m*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_18/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_18/beta/m
�
6Adam/batch_normalization_18/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/m*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_18/relux/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_18/relux/m
�
4Adam/activation_quant_18/relux/m/Read/ReadVariableOpReadVariableOp Adam/activation_quant_18/relux/m*
_output_shapes
: *
dtype0
�
Adam/dense_noise/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
**
shared_nameAdam/dense_noise/kernel/m
�
-Adam/dense_noise/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_noise/kernel/m*
_output_shapes

:@
*
dtype0
�
Adam/dense_noise/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_noise/bias/m

+Adam/dense_noise/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_noise/bias/m*
_output_shapes
:
*
dtype0
�
 Adam/activation_quant_15/relux/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_15/relux/v
�
4Adam/activation_quant_15/relux/v/Read/ReadVariableOpReadVariableOp Adam/activation_quant_15/relux/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_18/kernel/v
�
1Adam/conv2d_noise_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_18/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_18/bias/v
�
/Adam/conv2d_noise_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_18/bias/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_16/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_16/gamma/v
�
7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_16/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_16/beta/v
�
6Adam/batch_normalization_16/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/v*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_16/relux/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_16/relux/v
�
4Adam/activation_quant_16/relux/v/Read/ReadVariableOpReadVariableOp Adam/activation_quant_16/relux/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_19/kernel/v
�
1Adam/conv2d_noise_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_19/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_19/bias/v
�
/Adam/conv2d_noise_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_19/bias/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_17/gamma/v
�
7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_17/beta/v
�
6Adam/batch_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/v*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_17/relux/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_17/relux/v
�
4Adam/activation_quant_17/relux/v/Read/ReadVariableOpReadVariableOp Adam/activation_quant_17/relux/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_noise_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*.
shared_nameAdam/conv2d_noise_20/kernel/v
�
1Adam/conv2d_noise_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_20/kernel/v*&
_output_shapes
:@@*
dtype0
�
Adam/conv2d_noise_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameAdam/conv2d_noise_20/bias/v
�
/Adam/conv2d_noise_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_noise_20/bias/v*
_output_shapes
:@*
dtype0
�
#Adam/batch_normalization_18/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_18/gamma/v
�
7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/v*
_output_shapes
:@*
dtype0
�
"Adam/batch_normalization_18/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_18/beta/v
�
6Adam/batch_normalization_18/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/v*
_output_shapes
:@*
dtype0
�
 Adam/activation_quant_18/relux/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" Adam/activation_quant_18/relux/v
�
4Adam/activation_quant_18/relux/v/Read/ReadVariableOpReadVariableOp Adam/activation_quant_18/relux/v*
_output_shapes
: *
dtype0
�
Adam/dense_noise/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
**
shared_nameAdam/dense_noise/kernel/v
�
-Adam/dense_noise/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_noise/kernel/v*
_output_shapes

:@
*
dtype0
�
Adam/dense_noise/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_noise/bias/v

+Adam/dense_noise/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_noise/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
�s
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�r
value�rB�r B�r
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer-12
layer-13
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
]
	relux
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
 
�
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(trainable_variables
)regularization_losses
*	variables
+	keras_api
R
,trainable_variables
-regularization_losses
.	variables
/	keras_api
]
	0relux
1trainable_variables
2regularization_losses
3	variables
4	keras_api
h

5kernel
6bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
�
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
]
	Drelux
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
h

Ikernel
Jbias
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
�
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
R
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
R
\trainable_variables
]regularization_losses
^	variables
_	keras_api
R
`trainable_variables
aregularization_losses
b	variables
c	keras_api
]
	drelux
etrainable_variables
fregularization_losses
g	variables
h	keras_api
h

ikernel
jbias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
�
oiter

pbeta_1

qbeta_2
	rdecay
slearning_ratem�m�m�$m�%m�0m�5m�6m�<m�=m�Dm�Im�Jm�Pm�Qm�dm�im�jm�v�v�v�$v�%v�0v�5v�6v�<v�=v�Dv�Iv�Jv�Pv�Qv�dv�iv�jv�
�
0
1
2
$3
%4
05
56
67
<8
=9
D10
I11
J12
P13
Q14
d15
i16
j17
 
�
0
1
2
$3
%4
&5
'6
07
58
69
<10
=11
>12
?13
D14
I15
J16
P17
Q18
R19
S20
d21
i22
j23
�
tnon_trainable_variables
trainable_variables
umetrics
regularization_losses
vlayer_regularization_losses

wlayers
	variables
 
db
VARIABLE_VALUEactivation_quant_15/relux5layer_with_weights-0/relux/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
�
xnon_trainable_variables
trainable_variables
ymetrics
regularization_losses
zlayer_regularization_losses

{layers
	variables
b`
VARIABLE_VALUEconv2d_noise_18/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_noise_18/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
|non_trainable_variables
trainable_variables
}metrics
 regularization_losses
~layer_regularization_losses

layers
!	variables
 
ge
VARIABLE_VALUEbatch_normalization_16/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_16/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_16/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_16/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
&2
'3
�
�non_trainable_variables
(trainable_variables
�metrics
)regularization_losses
 �layer_regularization_losses
�layers
*	variables
 
 
 
�
�non_trainable_variables
,trainable_variables
�metrics
-regularization_losses
 �layer_regularization_losses
�layers
.	variables
db
VARIABLE_VALUEactivation_quant_16/relux5layer_with_weights-3/relux/.ATTRIBUTES/VARIABLE_VALUE

00
 

00
�
�non_trainable_variables
1trainable_variables
�metrics
2regularization_losses
 �layer_regularization_losses
�layers
3	variables
b`
VARIABLE_VALUEconv2d_noise_19/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_noise_19/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61
 

50
61
�
�non_trainable_variables
7trainable_variables
�metrics
8regularization_losses
 �layer_regularization_losses
�layers
9	variables
 
ge
VARIABLE_VALUEbatch_normalization_17/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_17/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_17/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_17/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
 

<0
=1
>2
?3
�
�non_trainable_variables
@trainable_variables
�metrics
Aregularization_losses
 �layer_regularization_losses
�layers
B	variables
db
VARIABLE_VALUEactivation_quant_17/relux5layer_with_weights-6/relux/.ATTRIBUTES/VARIABLE_VALUE

D0
 

D0
�
�non_trainable_variables
Etrainable_variables
�metrics
Fregularization_losses
 �layer_regularization_losses
�layers
G	variables
b`
VARIABLE_VALUEconv2d_noise_20/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEconv2d_noise_20/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1
 

I0
J1
�
�non_trainable_variables
Ktrainable_variables
�metrics
Lregularization_losses
 �layer_regularization_losses
�layers
M	variables
 
ge
VARIABLE_VALUEbatch_normalization_18/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_18/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_18/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_18/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1
 

P0
Q1
R2
S3
�
�non_trainable_variables
Ttrainable_variables
�metrics
Uregularization_losses
 �layer_regularization_losses
�layers
V	variables
 
 
 
�
�non_trainable_variables
Xtrainable_variables
�metrics
Yregularization_losses
 �layer_regularization_losses
�layers
Z	variables
 
 
 
�
�non_trainable_variables
\trainable_variables
�metrics
]regularization_losses
 �layer_regularization_losses
�layers
^	variables
 
 
 
�
�non_trainable_variables
`trainable_variables
�metrics
aregularization_losses
 �layer_regularization_losses
�layers
b	variables
db
VARIABLE_VALUEactivation_quant_18/relux5layer_with_weights-9/relux/.ATTRIBUTES/VARIABLE_VALUE

d0
 

d0
�
�non_trainable_variables
etrainable_variables
�metrics
fregularization_losses
 �layer_regularization_losses
�layers
g	variables
_]
VARIABLE_VALUEdense_noise/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdense_noise/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

i0
j1
 

i0
j1
�
�non_trainable_variables
ktrainable_variables
�metrics
lregularization_losses
 �layer_regularization_losses
�layers
m	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
*
&0
'1
>2
?3
R4
S5

�0
 
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
 
 
 
 
 
 
 
 

&0
'1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

>0
?1
 
 
 
 
 
 
 
 
 
 
 

R0
S1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


�total

�count
�
_fn_kwargs
�trainable_variables
�regularization_losses
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

�0
�1
�
�non_trainable_variables
�trainable_variables
�metrics
�regularization_losses
 �layer_regularization_losses
�layers
�	variables

�0
�1
 
 
 
��
VARIABLE_VALUE Adam/activation_quant_15/relux/mQlayer_with_weights-0/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_18/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/conv2d_noise_18/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_16/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_16/relux/mQlayer_with_weights-3/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_19/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/conv2d_noise_19/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_17/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_17/relux/mQlayer_with_weights-6/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_20/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/conv2d_noise_20/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_18/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_18/relux/mQlayer_with_weights-9/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_noise/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_noise/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_15/relux/vQlayer_with_weights-0/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_18/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/conv2d_noise_18/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_16/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_16/relux/vQlayer_with_weights-3/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_19/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/conv2d_noise_19/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_17/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_17/relux/vQlayer_with_weights-6/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/conv2d_noise_20/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUEAdam/conv2d_noise_20/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/batch_normalization_18/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/activation_quant_18/relux/vQlayer_with_weights-9/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_noise/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_noise/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*/
_output_shapes
:���������@*
dtype0*$
shape:���������@
�
serving_default_input_2Placeholder*/
_output_shapes
:���������@*
dtype0*$
shape:���������@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2activation_quant_15/reluxconv2d_noise_18/kernelconv2d_noise_18/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceactivation_quant_16/reluxconv2d_noise_19/kernelconv2d_noise_19/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceactivation_quant_17/reluxconv2d_noise_20/kernelconv2d_noise_20/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceactivation_quant_18/reluxdense_noise/kerneldense_noise/bias*%
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*-
f(R&
$__inference_signature_wrapper_119242
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-activation_quant_15/relux/Read/ReadVariableOp*conv2d_noise_18/kernel/Read/ReadVariableOp(conv2d_noise_18/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp-activation_quant_16/relux/Read/ReadVariableOp*conv2d_noise_19/kernel/Read/ReadVariableOp(conv2d_noise_19/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp-activation_quant_17/relux/Read/ReadVariableOp*conv2d_noise_20/kernel/Read/ReadVariableOp(conv2d_noise_20/bias/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp-activation_quant_18/relux/Read/ReadVariableOp&dense_noise/kernel/Read/ReadVariableOp$dense_noise/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/activation_quant_15/relux/m/Read/ReadVariableOp1Adam/conv2d_noise_18/kernel/m/Read/ReadVariableOp/Adam/conv2d_noise_18/bias/m/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_16/beta/m/Read/ReadVariableOp4Adam/activation_quant_16/relux/m/Read/ReadVariableOp1Adam/conv2d_noise_19/kernel/m/Read/ReadVariableOp/Adam/conv2d_noise_19/bias/m/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_17/beta/m/Read/ReadVariableOp4Adam/activation_quant_17/relux/m/Read/ReadVariableOp1Adam/conv2d_noise_20/kernel/m/Read/ReadVariableOp/Adam/conv2d_noise_20/bias/m/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_18/beta/m/Read/ReadVariableOp4Adam/activation_quant_18/relux/m/Read/ReadVariableOp-Adam/dense_noise/kernel/m/Read/ReadVariableOp+Adam/dense_noise/bias/m/Read/ReadVariableOp4Adam/activation_quant_15/relux/v/Read/ReadVariableOp1Adam/conv2d_noise_18/kernel/v/Read/ReadVariableOp/Adam/conv2d_noise_18/bias/v/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_16/beta/v/Read/ReadVariableOp4Adam/activation_quant_16/relux/v/Read/ReadVariableOp1Adam/conv2d_noise_19/kernel/v/Read/ReadVariableOp/Adam/conv2d_noise_19/bias/v/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_17/beta/v/Read/ReadVariableOp4Adam/activation_quant_17/relux/v/Read/ReadVariableOp1Adam/conv2d_noise_20/kernel/v/Read/ReadVariableOp/Adam/conv2d_noise_20/bias/v/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_18/beta/v/Read/ReadVariableOp4Adam/activation_quant_18/relux/v/Read/ReadVariableOp-Adam/dense_noise/kernel/v/Read/ReadVariableOp+Adam/dense_noise/bias/v/Read/ReadVariableOpConst*P
TinI
G2E	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: */
config_proto

CPU

GPU2*0,1J 8*(
f#R!
__inference__traced_save_120871
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameactivation_quant_15/reluxconv2d_noise_18/kernelconv2d_noise_18/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceactivation_quant_16/reluxconv2d_noise_19/kernelconv2d_noise_19/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceactivation_quant_17/reluxconv2d_noise_20/kernelconv2d_noise_20/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceactivation_quant_18/reluxdense_noise/kerneldense_noise/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount Adam/activation_quant_15/relux/mAdam/conv2d_noise_18/kernel/mAdam/conv2d_noise_18/bias/m#Adam/batch_normalization_16/gamma/m"Adam/batch_normalization_16/beta/m Adam/activation_quant_16/relux/mAdam/conv2d_noise_19/kernel/mAdam/conv2d_noise_19/bias/m#Adam/batch_normalization_17/gamma/m"Adam/batch_normalization_17/beta/m Adam/activation_quant_17/relux/mAdam/conv2d_noise_20/kernel/mAdam/conv2d_noise_20/bias/m#Adam/batch_normalization_18/gamma/m"Adam/batch_normalization_18/beta/m Adam/activation_quant_18/relux/mAdam/dense_noise/kernel/mAdam/dense_noise/bias/m Adam/activation_quant_15/relux/vAdam/conv2d_noise_18/kernel/vAdam/conv2d_noise_18/bias/v#Adam/batch_normalization_16/gamma/v"Adam/batch_normalization_16/beta/v Adam/activation_quant_16/relux/vAdam/conv2d_noise_19/kernel/vAdam/conv2d_noise_19/bias/v#Adam/batch_normalization_17/gamma/v"Adam/batch_normalization_17/beta/v Adam/activation_quant_17/relux/vAdam/conv2d_noise_20/kernel/vAdam/conv2d_noise_20/bias/v#Adam/batch_normalization_18/gamma/v"Adam/batch_normalization_18/beta/v Adam/activation_quant_18/relux/vAdam/dense_noise/kernel/vAdam/dense_noise/bias/v*O
TinH
F2D*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: */
config_proto

CPU

GPU2*0,1J 8*+
f&R$
"__inference__traced_restore_121084��
�
�
7__inference_batch_normalization_17_layer_call_fn_120149

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1184782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_118478

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_118463
assignmovingavg_1_118470
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/118463*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/118463*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_118463*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/118463*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/118463*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_118463AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/118463*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/118470*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/118470*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_118470*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/118470*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/118470*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_118470AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/118470*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
��
�
A__inference_model_layer_call_and_return_conditional_losses_118922
input_1
input_26
2activation_quant_15_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_19
5batch_normalization_16_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_39
5batch_normalization_16_statefulpartitionedcall_args_46
2activation_quant_16_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_19
5batch_normalization_17_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_39
5batch_normalization_17_statefulpartitionedcall_args_46
2activation_quant_17_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_19
5batch_normalization_18_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_39
5batch_normalization_18_statefulpartitionedcall_args_46
2activation_quant_18_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_2
identity��+activation_quant_15/StatefulPartitionedCall�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�+activation_quant_16/StatefulPartitionedCall�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�+activation_quant_17/StatefulPartitionedCall�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�+activation_quant_18/StatefulPartitionedCall�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�'conv2d_noise_18/StatefulPartitionedCall�'conv2d_noise_19/StatefulPartitionedCall�'conv2d_noise_20/StatefulPartitionedCall�#dense_noise/StatefulPartitionedCall�
+activation_quant_15/StatefulPartitionedCallStatefulPartitionedCallinput_22activation_quant_15_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_1181932-
+activation_quant_15/StatefulPartitionedCall�
'conv2d_noise_18/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_15/StatefulPartitionedCall:output:0.conv2d_noise_18_statefulpartitionedcall_args_1.conv2d_noise_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_1182432)
'conv2d_noise_18/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_18/StatefulPartitionedCall:output:05batch_normalization_16_statefulpartitionedcall_args_15batch_normalization_16_statefulpartitionedcall_args_25batch_normalization_16_statefulpartitionedcall_args_35batch_normalization_16_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_11831720
.batch_normalization_16/StatefulPartitionedCall�
add/PartitionedCallPartitionedCallinput_17batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1183472
add/PartitionedCall�
+activation_quant_16/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:02activation_quant_16_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_1183762-
+activation_quant_16/StatefulPartitionedCall�
'conv2d_noise_19/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:0.conv2d_noise_19_statefulpartitionedcall_args_1.conv2d_noise_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_1184262)
'conv2d_noise_19/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_19/StatefulPartitionedCall:output:05batch_normalization_17_statefulpartitionedcall_args_15batch_normalization_17_statefulpartitionedcall_args_25batch_normalization_17_statefulpartitionedcall_args_35batch_normalization_17_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_11850020
.batch_normalization_17/StatefulPartitionedCall�
+activation_quant_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:02activation_quant_17_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_1185442-
+activation_quant_17/StatefulPartitionedCall�
'conv2d_noise_20/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_17/StatefulPartitionedCall:output:0.conv2d_noise_20_statefulpartitionedcall_args_1.conv2d_noise_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_1185942)
'conv2d_noise_20/StatefulPartitionedCall�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_20/StatefulPartitionedCall:output:05batch_normalization_18_statefulpartitionedcall_args_15batch_normalization_18_statefulpartitionedcall_args_25batch_normalization_18_statefulpartitionedcall_args_35batch_normalization_18_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_11866820
.batch_normalization_18/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:07batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_1186982
add_1/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_1181622#
!average_pooling2d/PartitionedCall�
flatten/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1187142
flatten/PartitionedCall�
+activation_quant_18/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:02activation_quant_18_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_1187422-
+activation_quant_18/StatefulPartitionedCall�
#dense_noise/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_18/StatefulPartitionedCall:output:0*dense_noise_statefulpartitionedcall_args_1*dense_noise_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*P
fKRI
G__inference_dense_noise_layer_call_and_return_conditional_losses_1187942%
#dense_noise/StatefulPartitionedCall�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_15_statefulpartitionedcall_args_1,^activation_quant_15/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_16_statefulpartitionedcall_args_1,^activation_quant_16/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_17_statefulpartitionedcall_args_1,^activation_quant_17/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_18_statefulpartitionedcall_args_1,^activation_quant_18/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentity,dense_noise/StatefulPartitionedCall:output:0,^activation_quant_15/StatefulPartitionedCall<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp,^activation_quant_16/StatefulPartitionedCall<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp,^activation_quant_17/StatefulPartitionedCall<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp,^activation_quant_18/StatefulPartitionedCall<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall(^conv2d_noise_18/StatefulPartitionedCall(^conv2d_noise_19/StatefulPartitionedCall(^conv2d_noise_20/StatefulPartitionedCall$^dense_noise/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@::::::::::::::::::::::::2Z
+activation_quant_15/StatefulPartitionedCall+activation_quant_15/StatefulPartitionedCall2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_16/StatefulPartitionedCall+activation_quant_16/StatefulPartitionedCall2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_17/StatefulPartitionedCall+activation_quant_17/StatefulPartitionedCall2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_18/StatefulPartitionedCall+activation_quant_18/StatefulPartitionedCall2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2R
'conv2d_noise_18/StatefulPartitionedCall'conv2d_noise_18/StatefulPartitionedCall2R
'conv2d_noise_19/StatefulPartitionedCall'conv2d_noise_19/StatefulPartitionedCall2R
'conv2d_noise_20/StatefulPartitionedCall'conv2d_noise_20/StatefulPartitionedCall2J
#dense_noise/StatefulPartitionedCall#dense_noise/StatefulPartitionedCall:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
�$
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_120440

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_120425
assignmovingavg_1_120432
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/120425*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/120425*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_120425*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/120425*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/120425*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_120425AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/120425*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/120432*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/120432*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_120432*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/120432*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/120432*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_120432AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/120432*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_118149

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_120632H
Dactivation_quant_17_relux_regularizer_square_readvariableop_resource
identity��;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOpDactivation_quant_17_relux_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
IdentityIdentity-activation_quant_17/relux/Regularizer/add:z:0<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp
�
�
7__inference_batch_normalization_16_layer_call_fn_119963

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1178542
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�}
� 
__inference__traced_save_120871
file_prefix8
4savev2_activation_quant_15_relux_read_readvariableop5
1savev2_conv2d_noise_18_kernel_read_readvariableop3
/savev2_conv2d_noise_18_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableop8
4savev2_activation_quant_16_relux_read_readvariableop5
1savev2_conv2d_noise_19_kernel_read_readvariableop3
/savev2_conv2d_noise_19_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableop8
4savev2_activation_quant_17_relux_read_readvariableop5
1savev2_conv2d_noise_20_kernel_read_readvariableop3
/savev2_conv2d_noise_20_bias_read_readvariableop;
7savev2_batch_normalization_18_gamma_read_readvariableop:
6savev2_batch_normalization_18_beta_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableop8
4savev2_activation_quant_18_relux_read_readvariableop1
-savev2_dense_noise_kernel_read_readvariableop/
+savev2_dense_noise_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adam_activation_quant_15_relux_m_read_readvariableop<
8savev2_adam_conv2d_noise_18_kernel_m_read_readvariableop:
6savev2_adam_conv2d_noise_18_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_m_read_readvariableop?
;savev2_adam_activation_quant_16_relux_m_read_readvariableop<
8savev2_adam_conv2d_noise_19_kernel_m_read_readvariableop:
6savev2_adam_conv2d_noise_19_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_m_read_readvariableop?
;savev2_adam_activation_quant_17_relux_m_read_readvariableop<
8savev2_adam_conv2d_noise_20_kernel_m_read_readvariableop:
6savev2_adam_conv2d_noise_20_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_m_read_readvariableop?
;savev2_adam_activation_quant_18_relux_m_read_readvariableop8
4savev2_adam_dense_noise_kernel_m_read_readvariableop6
2savev2_adam_dense_noise_bias_m_read_readvariableop?
;savev2_adam_activation_quant_15_relux_v_read_readvariableop<
8savev2_adam_conv2d_noise_18_kernel_v_read_readvariableop:
6savev2_adam_conv2d_noise_18_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_v_read_readvariableop?
;savev2_adam_activation_quant_16_relux_v_read_readvariableop<
8savev2_adam_conv2d_noise_19_kernel_v_read_readvariableop:
6savev2_adam_conv2d_noise_19_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_v_read_readvariableop?
;savev2_adam_activation_quant_17_relux_v_read_readvariableop<
8savev2_adam_conv2d_noise_20_kernel_v_read_readvariableop:
6savev2_adam_conv2d_noise_20_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_v_read_readvariableop?
;savev2_adam_activation_quant_18_relux_v_read_readvariableop8
4savev2_adam_dense_noise_kernel_v_read_readvariableop6
2savev2_adam_dense_noise_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_07ba154d27ee4f148525a7abdfb67443/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*�$
value�$B�$CB5layer_with_weights-0/relux/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/relux/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/relux/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/relux/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*�
value�B�CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_activation_quant_15_relux_read_readvariableop1savev2_conv2d_noise_18_kernel_read_readvariableop/savev2_conv2d_noise_18_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop4savev2_activation_quant_16_relux_read_readvariableop1savev2_conv2d_noise_19_kernel_read_readvariableop/savev2_conv2d_noise_19_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop4savev2_activation_quant_17_relux_read_readvariableop1savev2_conv2d_noise_20_kernel_read_readvariableop/savev2_conv2d_noise_20_bias_read_readvariableop7savev2_batch_normalization_18_gamma_read_readvariableop6savev2_batch_normalization_18_beta_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop4savev2_activation_quant_18_relux_read_readvariableop-savev2_dense_noise_kernel_read_readvariableop+savev2_dense_noise_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_activation_quant_15_relux_m_read_readvariableop8savev2_adam_conv2d_noise_18_kernel_m_read_readvariableop6savev2_adam_conv2d_noise_18_bias_m_read_readvariableop>savev2_adam_batch_normalization_16_gamma_m_read_readvariableop=savev2_adam_batch_normalization_16_beta_m_read_readvariableop;savev2_adam_activation_quant_16_relux_m_read_readvariableop8savev2_adam_conv2d_noise_19_kernel_m_read_readvariableop6savev2_adam_conv2d_noise_19_bias_m_read_readvariableop>savev2_adam_batch_normalization_17_gamma_m_read_readvariableop=savev2_adam_batch_normalization_17_beta_m_read_readvariableop;savev2_adam_activation_quant_17_relux_m_read_readvariableop8savev2_adam_conv2d_noise_20_kernel_m_read_readvariableop6savev2_adam_conv2d_noise_20_bias_m_read_readvariableop>savev2_adam_batch_normalization_18_gamma_m_read_readvariableop=savev2_adam_batch_normalization_18_beta_m_read_readvariableop;savev2_adam_activation_quant_18_relux_m_read_readvariableop4savev2_adam_dense_noise_kernel_m_read_readvariableop2savev2_adam_dense_noise_bias_m_read_readvariableop;savev2_adam_activation_quant_15_relux_v_read_readvariableop8savev2_adam_conv2d_noise_18_kernel_v_read_readvariableop6savev2_adam_conv2d_noise_18_bias_v_read_readvariableop>savev2_adam_batch_normalization_16_gamma_v_read_readvariableop=savev2_adam_batch_normalization_16_beta_v_read_readvariableop;savev2_adam_activation_quant_16_relux_v_read_readvariableop8savev2_adam_conv2d_noise_19_kernel_v_read_readvariableop6savev2_adam_conv2d_noise_19_bias_v_read_readvariableop>savev2_adam_batch_normalization_17_gamma_v_read_readvariableop=savev2_adam_batch_normalization_17_beta_v_read_readvariableop;savev2_adam_activation_quant_17_relux_v_read_readvariableop8savev2_adam_conv2d_noise_20_kernel_v_read_readvariableop6savev2_adam_conv2d_noise_20_bias_v_read_readvariableop>savev2_adam_batch_normalization_18_gamma_v_read_readvariableop=savev2_adam_batch_normalization_18_beta_v_read_readvariableop;savev2_adam_activation_quant_18_relux_v_read_readvariableop4savev2_adam_dense_noise_kernel_v_read_readvariableop2savev2_adam_dense_noise_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *Q
dtypesG
E2C	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : :@@:@:@:@:@:@: :@@:@:@:@:@:@: :@@:@:@:@:@:@: :@
:
: : : : : : : : :@@:@:@:@: :@@:@:@:@: :@@:@:@:@: :@
:
: :@@:@:@:@: :@@:@:@:@: :@@:@:@:@: :@
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
�
7__inference_batch_normalization_18_layer_call_fn_120397

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1181182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_118233
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
�
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_120260
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_120498

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
G__inference_dense_noise_layer_call_and_return_conditional_losses_120568
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*
_output_shapes

:@
*
dtype02
Abs/ReadVariableOpV
AbsAbsAbs/ReadVariableOp:value:0*
T0*
_output_shapes

:@
2
Abs_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul{
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   
   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:@
*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*
_output_shapes

:@
2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes

:@
2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*
_output_shapes

:@
*
dtype02
ReadVariableOpg
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*
_output_shapes

:@
2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:
*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:
2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:
2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:
2
add_1X
MatMulMatMulxadd:z:0*
T0*'
_output_shapes
:���������
2
MatMulf
add_2AddV2MatMul:product:0	add_1:z:0*
T0*'
_output_shapes
:���������
2
add_2Z
SoftmaxSoftmax	add_2:z:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�$
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_118646

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_118631
assignmovingavg_1_118638
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/118631*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/118631*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_118631*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/118631*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/118631*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_118631AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/118631*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/118638*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/118638*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_118638*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/118638*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/118638*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_118638AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/118638*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
0__inference_conv2d_noise_18_layer_call_fn_119812
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_1182432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�$
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_117854

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_117839
assignmovingavg_1_117846
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/117839*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/117839*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_117839*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/117839*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/117839*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_117839AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/117839*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/117846*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/117846*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_117846*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/117846*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/117846*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_117846AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/117846*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
0__inference_conv2d_noise_20_layer_call_fn_120320
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_1185942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_119954

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
D
(__inference_flatten_layer_call_fn_120503

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1187142
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_117885

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_118295

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_118280
assignmovingavg_1_118287
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/118280*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/118280*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_118280*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/118280*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/118280*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_118280AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/118280*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/118287*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/118287*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_118287*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/118287*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/118287*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_118287AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/118287*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_120606H
Dactivation_quant_15_relux_regularizer_square_readvariableop_resource
identity��;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOpDactivation_quant_15_relux_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
IdentityIdentity-activation_quant_15/relux/Regularizer/add:z:0<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp
�$
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_120366

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_120351
assignmovingavg_1_120358
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/120351*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/120351*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_120351*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/120351*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/120351*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_120351AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/120351*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/120358*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/120358*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_120358*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/120358*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/120358*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_120358AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/120358*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�'
"__inference__traced_restore_121084
file_prefix.
*assignvariableop_activation_quant_15_relux-
)assignvariableop_1_conv2d_noise_18_kernel+
'assignvariableop_2_conv2d_noise_18_bias3
/assignvariableop_3_batch_normalization_16_gamma2
.assignvariableop_4_batch_normalization_16_beta9
5assignvariableop_5_batch_normalization_16_moving_mean=
9assignvariableop_6_batch_normalization_16_moving_variance0
,assignvariableop_7_activation_quant_16_relux-
)assignvariableop_8_conv2d_noise_19_kernel+
'assignvariableop_9_conv2d_noise_19_bias4
0assignvariableop_10_batch_normalization_17_gamma3
/assignvariableop_11_batch_normalization_17_beta:
6assignvariableop_12_batch_normalization_17_moving_mean>
:assignvariableop_13_batch_normalization_17_moving_variance1
-assignvariableop_14_activation_quant_17_relux.
*assignvariableop_15_conv2d_noise_20_kernel,
(assignvariableop_16_conv2d_noise_20_bias4
0assignvariableop_17_batch_normalization_18_gamma3
/assignvariableop_18_batch_normalization_18_beta:
6assignvariableop_19_batch_normalization_18_moving_mean>
:assignvariableop_20_batch_normalization_18_moving_variance1
-assignvariableop_21_activation_quant_18_relux*
&assignvariableop_22_dense_noise_kernel(
$assignvariableop_23_dense_noise_bias!
assignvariableop_24_adam_iter#
assignvariableop_25_adam_beta_1#
assignvariableop_26_adam_beta_2"
assignvariableop_27_adam_decay*
&assignvariableop_28_adam_learning_rate
assignvariableop_29_total
assignvariableop_30_count8
4assignvariableop_31_adam_activation_quant_15_relux_m5
1assignvariableop_32_adam_conv2d_noise_18_kernel_m3
/assignvariableop_33_adam_conv2d_noise_18_bias_m;
7assignvariableop_34_adam_batch_normalization_16_gamma_m:
6assignvariableop_35_adam_batch_normalization_16_beta_m8
4assignvariableop_36_adam_activation_quant_16_relux_m5
1assignvariableop_37_adam_conv2d_noise_19_kernel_m3
/assignvariableop_38_adam_conv2d_noise_19_bias_m;
7assignvariableop_39_adam_batch_normalization_17_gamma_m:
6assignvariableop_40_adam_batch_normalization_17_beta_m8
4assignvariableop_41_adam_activation_quant_17_relux_m5
1assignvariableop_42_adam_conv2d_noise_20_kernel_m3
/assignvariableop_43_adam_conv2d_noise_20_bias_m;
7assignvariableop_44_adam_batch_normalization_18_gamma_m:
6assignvariableop_45_adam_batch_normalization_18_beta_m8
4assignvariableop_46_adam_activation_quant_18_relux_m1
-assignvariableop_47_adam_dense_noise_kernel_m/
+assignvariableop_48_adam_dense_noise_bias_m8
4assignvariableop_49_adam_activation_quant_15_relux_v5
1assignvariableop_50_adam_conv2d_noise_18_kernel_v3
/assignvariableop_51_adam_conv2d_noise_18_bias_v;
7assignvariableop_52_adam_batch_normalization_16_gamma_v:
6assignvariableop_53_adam_batch_normalization_16_beta_v8
4assignvariableop_54_adam_activation_quant_16_relux_v5
1assignvariableop_55_adam_conv2d_noise_19_kernel_v3
/assignvariableop_56_adam_conv2d_noise_19_bias_v;
7assignvariableop_57_adam_batch_normalization_17_gamma_v:
6assignvariableop_58_adam_batch_normalization_17_beta_v8
4assignvariableop_59_adam_activation_quant_17_relux_v5
1assignvariableop_60_adam_conv2d_noise_20_kernel_v3
/assignvariableop_61_adam_conv2d_noise_20_bias_v;
7assignvariableop_62_adam_batch_normalization_18_gamma_v:
6assignvariableop_63_adam_batch_normalization_18_beta_v8
4assignvariableop_64_adam_activation_quant_18_relux_v1
-assignvariableop_65_adam_dense_noise_kernel_v/
+assignvariableop_66_adam_dense_noise_bias_v
identity_68��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*�$
value�$B�$CB5layer_with_weights-0/relux/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/relux/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/relux/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/relux/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/relux/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/relux/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*�
value�B�CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Q
dtypesG
E2C	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp*assignvariableop_activation_quant_15_reluxIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp)assignvariableop_1_conv2d_noise_18_kernelIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp'assignvariableop_2_conv2d_noise_18_biasIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp/assignvariableop_3_batch_normalization_16_gammaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_16_betaIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp5assignvariableop_5_batch_normalization_16_moving_meanIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp9assignvariableop_6_batch_normalization_16_moving_varianceIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp,assignvariableop_7_activation_quant_16_reluxIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp)assignvariableop_8_conv2d_noise_19_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp'assignvariableop_9_conv2d_noise_19_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp0assignvariableop_10_batch_normalization_17_gammaIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_17_betaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp6assignvariableop_12_batch_normalization_17_moving_meanIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp:assignvariableop_13_batch_normalization_17_moving_varianceIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp-assignvariableop_14_activation_quant_17_reluxIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_conv2d_noise_20_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_conv2d_noise_20_biasIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp0assignvariableop_17_batch_normalization_18_gammaIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_18_betaIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp6assignvariableop_19_batch_normalization_18_moving_meanIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp:assignvariableop_20_batch_normalization_18_moving_varianceIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp-assignvariableop_21_activation_quant_18_reluxIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp&assignvariableop_22_dense_noise_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp$assignvariableop_23_dense_noise_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0	*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_1Identity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_2Identity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_decayIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_learning_rateIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_activation_quant_15_relux_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp1assignvariableop_32_adam_conv2d_noise_18_kernel_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp/assignvariableop_33_adam_conv2d_noise_18_bias_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp7assignvariableop_34_adam_batch_normalization_16_gamma_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_batch_normalization_16_beta_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp4assignvariableop_36_adam_activation_quant_16_relux_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp1assignvariableop_37_adam_conv2d_noise_19_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp/assignvariableop_38_adam_conv2d_noise_19_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp7assignvariableop_39_adam_batch_normalization_17_gamma_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adam_batch_normalization_17_beta_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp4assignvariableop_41_adam_activation_quant_17_relux_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp1assignvariableop_42_adam_conv2d_noise_20_kernel_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp/assignvariableop_43_adam_conv2d_noise_20_bias_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp7assignvariableop_44_adam_batch_normalization_18_gamma_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_batch_normalization_18_beta_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp4assignvariableop_46_adam_activation_quant_18_relux_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp-assignvariableop_47_adam_dense_noise_kernel_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_dense_noise_bias_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp4assignvariableop_49_adam_activation_quant_15_relux_vIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp1assignvariableop_50_adam_conv2d_noise_18_kernel_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp/assignvariableop_51_adam_conv2d_noise_18_bias_vIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp7assignvariableop_52_adam_batch_normalization_16_gamma_vIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_16_beta_vIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp4assignvariableop_54_adam_activation_quant_16_relux_vIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp1assignvariableop_55_adam_conv2d_noise_19_kernel_vIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp/assignvariableop_56_adam_conv2d_noise_19_bias_vIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_batch_normalization_17_gamma_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adam_batch_normalization_17_beta_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp4assignvariableop_59_adam_activation_quant_17_relux_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp1assignvariableop_60_adam_conv2d_noise_20_kernel_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp/assignvariableop_61_adam_conv2d_noise_20_bias_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp7assignvariableop_62_adam_batch_normalization_18_gamma_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp6assignvariableop_63_adam_batch_normalization_18_beta_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp4assignvariableop_64_adam_activation_quant_18_relux_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp-assignvariableop_65_adam_dense_noise_kernel_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp+assignvariableop_66_adam_dense_noise_bias_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_67�
Identity_68IdentityIdentity_67:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_68"#
identity_68Identity_68:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
��
�
A__inference_model_layer_call_and_return_conditional_losses_119002

inputs
inputs_16
2activation_quant_15_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_19
5batch_normalization_16_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_39
5batch_normalization_16_statefulpartitionedcall_args_46
2activation_quant_16_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_19
5batch_normalization_17_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_39
5batch_normalization_17_statefulpartitionedcall_args_46
2activation_quant_17_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_19
5batch_normalization_18_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_39
5batch_normalization_18_statefulpartitionedcall_args_46
2activation_quant_18_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_2
identity��+activation_quant_15/StatefulPartitionedCall�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�+activation_quant_16/StatefulPartitionedCall�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�+activation_quant_17/StatefulPartitionedCall�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�+activation_quant_18/StatefulPartitionedCall�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�'conv2d_noise_18/StatefulPartitionedCall�'conv2d_noise_19/StatefulPartitionedCall�'conv2d_noise_20/StatefulPartitionedCall�#dense_noise/StatefulPartitionedCall�
+activation_quant_15/StatefulPartitionedCallStatefulPartitionedCallinputs_12activation_quant_15_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_1181932-
+activation_quant_15/StatefulPartitionedCall�
'conv2d_noise_18/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_15/StatefulPartitionedCall:output:0.conv2d_noise_18_statefulpartitionedcall_args_1.conv2d_noise_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_1182332)
'conv2d_noise_18/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_18/StatefulPartitionedCall:output:05batch_normalization_16_statefulpartitionedcall_args_15batch_normalization_16_statefulpartitionedcall_args_25batch_normalization_16_statefulpartitionedcall_args_35batch_normalization_16_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_11829520
.batch_normalization_16/StatefulPartitionedCall�
add/PartitionedCallPartitionedCallinputs7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1183472
add/PartitionedCall�
+activation_quant_16/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:02activation_quant_16_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_1183762-
+activation_quant_16/StatefulPartitionedCall�
'conv2d_noise_19/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:0.conv2d_noise_19_statefulpartitionedcall_args_1.conv2d_noise_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_1184162)
'conv2d_noise_19/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_19/StatefulPartitionedCall:output:05batch_normalization_17_statefulpartitionedcall_args_15batch_normalization_17_statefulpartitionedcall_args_25batch_normalization_17_statefulpartitionedcall_args_35batch_normalization_17_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_11847820
.batch_normalization_17/StatefulPartitionedCall�
+activation_quant_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:02activation_quant_17_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_1185442-
+activation_quant_17/StatefulPartitionedCall�
'conv2d_noise_20/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_17/StatefulPartitionedCall:output:0.conv2d_noise_20_statefulpartitionedcall_args_1.conv2d_noise_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_1185842)
'conv2d_noise_20/StatefulPartitionedCall�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_20/StatefulPartitionedCall:output:05batch_normalization_18_statefulpartitionedcall_args_15batch_normalization_18_statefulpartitionedcall_args_25batch_normalization_18_statefulpartitionedcall_args_35batch_normalization_18_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_11864620
.batch_normalization_18/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:07batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_1186982
add_1/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_1181622#
!average_pooling2d/PartitionedCall�
flatten/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1187142
flatten/PartitionedCall�
+activation_quant_18/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:02activation_quant_18_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_1187422-
+activation_quant_18/StatefulPartitionedCall�
#dense_noise/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_18/StatefulPartitionedCall:output:0*dense_noise_statefulpartitionedcall_args_1*dense_noise_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*P
fKRI
G__inference_dense_noise_layer_call_and_return_conditional_losses_1187832%
#dense_noise/StatefulPartitionedCall�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_15_statefulpartitionedcall_args_1,^activation_quant_15/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_16_statefulpartitionedcall_args_1,^activation_quant_16/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_17_statefulpartitionedcall_args_1,^activation_quant_17/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_18_statefulpartitionedcall_args_1,^activation_quant_18/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentity,dense_noise/StatefulPartitionedCall:output:0,^activation_quant_15/StatefulPartitionedCall<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp,^activation_quant_16/StatefulPartitionedCall<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp,^activation_quant_17/StatefulPartitionedCall<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp,^activation_quant_18/StatefulPartitionedCall<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall(^conv2d_noise_18/StatefulPartitionedCall(^conv2d_noise_19/StatefulPartitionedCall(^conv2d_noise_20/StatefulPartitionedCall$^dense_noise/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@::::::::::::::::::::::::2Z
+activation_quant_15/StatefulPartitionedCall+activation_quant_15/StatefulPartitionedCall2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_16/StatefulPartitionedCall+activation_quant_16/StatefulPartitionedCall2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_17/StatefulPartitionedCall+activation_quant_17/StatefulPartitionedCall2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_18/StatefulPartitionedCall+activation_quant_18/StatefulPartitionedCall2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2R
'conv2d_noise_18/StatefulPartitionedCall'conv2d_noise_18/StatefulPartitionedCall2R
'conv2d_noise_19/StatefulPartitionedCall'conv2d_noise_19/StatefulPartitionedCall2R
'conv2d_noise_20/StatefulPartitionedCall'conv2d_noise_20/StatefulPartitionedCall2J
#dense_noise/StatefulPartitionedCall#dense_noise/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_120192

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_120177
assignmovingavg_1_120184
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/120177*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/120177*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_120177*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/120177*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/120177*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_120177AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/120177*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/120184*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/120184*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_120184*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/120184*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/120184*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_120184AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/120184*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
N
2__inference_average_pooling2d_layer_call_fn_118168

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_1181622
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
0__inference_conv2d_noise_19_layer_call_fn_120065
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_1184162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_118742
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpr
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yp
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*'
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*'
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0**
_input_shapes
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
��
�
A__inference_model_layer_call_and_return_conditional_losses_119664
inputs_0
inputs_17
3activation_quant_15_minimum_readvariableop_resource7
3conv2d_noise_18_convolution_readvariableop_resource3
/conv2d_noise_18_biasadd_readvariableop_resource2
.batch_normalization_16_readvariableop_resource4
0batch_normalization_16_readvariableop_1_resourceC
?batch_normalization_16_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource7
3activation_quant_16_minimum_readvariableop_resource7
3conv2d_noise_19_convolution_readvariableop_resource3
/conv2d_noise_19_biasadd_readvariableop_resource2
.batch_normalization_17_readvariableop_resource4
0batch_normalization_17_readvariableop_1_resourceC
?batch_normalization_17_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource7
3activation_quant_17_minimum_readvariableop_resource7
3conv2d_noise_20_convolution_readvariableop_resource3
/conv2d_noise_20_biasadd_readvariableop_resource2
.batch_normalization_18_readvariableop_resource4
0batch_normalization_18_readvariableop_1_resourceC
?batch_normalization_18_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource7
3activation_quant_18_minimum_readvariableop_resource.
*dense_noise_matmul_readvariableop_resource+
'dense_noise_add_readvariableop_resource
identity��:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_15/Minimum/ReadVariableOp�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_16/Minimum/ReadVariableOp�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_17/Minimum/ReadVariableOp�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_18/Minimum/ReadVariableOp�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�6batch_normalization_16/FusedBatchNormV3/ReadVariableOp�8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_16/ReadVariableOp�'batch_normalization_16/ReadVariableOp_1�6batch_normalization_17/FusedBatchNormV3/ReadVariableOp�8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_17/ReadVariableOp�'batch_normalization_17/ReadVariableOp_1�6batch_normalization_18/FusedBatchNormV3/ReadVariableOp�8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_18/ReadVariableOp�'batch_normalization_18/ReadVariableOp_1�&conv2d_noise_18/BiasAdd/ReadVariableOp�*conv2d_noise_18/convolution/ReadVariableOp�&conv2d_noise_19/BiasAdd/ReadVariableOp�*conv2d_noise_19/convolution/ReadVariableOp�&conv2d_noise_20/BiasAdd/ReadVariableOp�*conv2d_noise_20/convolution/ReadVariableOp�!dense_noise/MatMul/ReadVariableOp�dense_noise/add/ReadVariableOp�
*activation_quant_15/Minimum/ReadVariableOpReadVariableOp3activation_quant_15_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_15/Minimum/ReadVariableOp�
activation_quant_15/MinimumMinimuminputs_12activation_quant_15/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_15/Minimum�
activation_quant_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_15/Maximum/y�
activation_quant_15/MaximumMaximumactivation_quant_15/Minimum:z:0&activation_quant_15/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_15/Maximum{
activation_quant_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_15/Const�
:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_15_minimum_readvariableop_resource+^activation_quant_15/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_15/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_15/Maximum:z:0"activation_quant_15/Const:output:0Bactivation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_15/FakeQuantWithMinMaxVars�
*conv2d_noise_18/convolution/ReadVariableOpReadVariableOp3conv2d_noise_18_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*conv2d_noise_18/convolution/ReadVariableOp�
conv2d_noise_18/convolutionConv2D5activation_quant_15/FakeQuantWithMinMaxVars:outputs:02conv2d_noise_18/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_18/convolution�
&conv2d_noise_18/BiasAdd/ReadVariableOpReadVariableOp/conv2d_noise_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&conv2d_noise_18/BiasAdd/ReadVariableOp�
conv2d_noise_18/BiasAddBiasAdd$conv2d_noise_18/convolution:output:0.conv2d_noise_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_18/BiasAdd�
#batch_normalization_16/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_16/LogicalAnd/x�
#batch_normalization_16/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_16/LogicalAnd/y�
!batch_normalization_16/LogicalAnd
LogicalAnd,batch_normalization_16/LogicalAnd/x:output:0,batch_normalization_16/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_16/LogicalAnd�
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_16/ReadVariableOp�
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_16/ReadVariableOp_1�
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_18/BiasAdd:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_16/FusedBatchNormV3�
batch_normalization_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_16/Const�
add/addAddV2inputs_0+batch_normalization_16/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2	
add/add�
*activation_quant_16/Minimum/ReadVariableOpReadVariableOp3activation_quant_16_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_16/Minimum/ReadVariableOp�
activation_quant_16/MinimumMinimumadd/add:z:02activation_quant_16/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_16/Minimum�
activation_quant_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_16/Maximum/y�
activation_quant_16/MaximumMaximumactivation_quant_16/Minimum:z:0&activation_quant_16/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_16/Maximum{
activation_quant_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_16/Const�
:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_16_minimum_readvariableop_resource+^activation_quant_16/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_16/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_16/Maximum:z:0"activation_quant_16/Const:output:0Bactivation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_16/FakeQuantWithMinMaxVars�
*conv2d_noise_19/convolution/ReadVariableOpReadVariableOp3conv2d_noise_19_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*conv2d_noise_19/convolution/ReadVariableOp�
conv2d_noise_19/convolutionConv2D5activation_quant_16/FakeQuantWithMinMaxVars:outputs:02conv2d_noise_19/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_19/convolution�
&conv2d_noise_19/BiasAdd/ReadVariableOpReadVariableOp/conv2d_noise_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&conv2d_noise_19/BiasAdd/ReadVariableOp�
conv2d_noise_19/BiasAddBiasAdd$conv2d_noise_19/convolution:output:0.conv2d_noise_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_19/BiasAdd�
#batch_normalization_17/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_17/LogicalAnd/x�
#batch_normalization_17/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_17/LogicalAnd/y�
!batch_normalization_17/LogicalAnd
LogicalAnd,batch_normalization_17/LogicalAnd/x:output:0,batch_normalization_17/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_17/LogicalAnd�
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_17/ReadVariableOp�
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_17/ReadVariableOp_1�
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_19/BiasAdd:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_17/FusedBatchNormV3�
batch_normalization_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_17/Const�
*activation_quant_17/Minimum/ReadVariableOpReadVariableOp3activation_quant_17_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_17/Minimum/ReadVariableOp�
activation_quant_17/MinimumMinimum+batch_normalization_17/FusedBatchNormV3:y:02activation_quant_17/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_17/Minimum�
activation_quant_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_17/Maximum/y�
activation_quant_17/MaximumMaximumactivation_quant_17/Minimum:z:0&activation_quant_17/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_17/Maximum{
activation_quant_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_17/Const�
:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_17_minimum_readvariableop_resource+^activation_quant_17/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_17/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_17/Maximum:z:0"activation_quant_17/Const:output:0Bactivation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_17/FakeQuantWithMinMaxVars�
*conv2d_noise_20/convolution/ReadVariableOpReadVariableOp3conv2d_noise_20_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*conv2d_noise_20/convolution/ReadVariableOp�
conv2d_noise_20/convolutionConv2D5activation_quant_17/FakeQuantWithMinMaxVars:outputs:02conv2d_noise_20/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_20/convolution�
&conv2d_noise_20/BiasAdd/ReadVariableOpReadVariableOp/conv2d_noise_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&conv2d_noise_20/BiasAdd/ReadVariableOp�
conv2d_noise_20/BiasAddBiasAdd$conv2d_noise_20/convolution:output:0.conv2d_noise_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_20/BiasAdd�
#batch_normalization_18/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#batch_normalization_18/LogicalAnd/x�
#batch_normalization_18/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_18/LogicalAnd/y�
!batch_normalization_18/LogicalAnd
LogicalAnd,batch_normalization_18/LogicalAnd/x:output:0,batch_normalization_18/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_18/LogicalAnd�
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_18/ReadVariableOp�
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_18/ReadVariableOp_1�
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_20/BiasAdd:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2)
'batch_normalization_18/FusedBatchNormV3�
batch_normalization_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_18/Const�
	add_1/addAddV25activation_quant_16/FakeQuantWithMinMaxVars:outputs:0+batch_normalization_18/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
	add_1/add�
average_pooling2d/AvgPoolAvgPooladd_1/add:z:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   2
flatten/Const�
flatten/ReshapeReshape"average_pooling2d/AvgPool:output:0flatten/Const:output:0*
T0*'
_output_shapes
:���������@2
flatten/Reshape�
*activation_quant_18/Minimum/ReadVariableOpReadVariableOp3activation_quant_18_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_18/Minimum/ReadVariableOp�
activation_quant_18/MinimumMinimumflatten/Reshape:output:02activation_quant_18/Minimum/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
activation_quant_18/Minimum�
activation_quant_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_18/Maximum/y�
activation_quant_18/MaximumMaximumactivation_quant_18/Minimum:z:0&activation_quant_18/Maximum/y:output:0*
T0*'
_output_shapes
:���������@2
activation_quant_18/Maximum{
activation_quant_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_18/Const�
:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_18_minimum_readvariableop_resource+^activation_quant_18/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_18/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_18/Maximum:z:0"activation_quant_18/Const:output:0Bactivation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*'
_output_shapes
:���������@*
num_bits2-
+activation_quant_18/FakeQuantWithMinMaxVars�
!dense_noise/MatMul/ReadVariableOpReadVariableOp*dense_noise_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02#
!dense_noise/MatMul/ReadVariableOp�
dense_noise/MatMulMatMul5activation_quant_18/FakeQuantWithMinMaxVars:outputs:0)dense_noise/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_noise/MatMul�
dense_noise/add/ReadVariableOpReadVariableOp'dense_noise_add_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_noise/add/ReadVariableOp�
dense_noise/addAddV2dense_noise/MatMul:product:0&dense_noise/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_noise/add|
dense_noise/SoftmaxSoftmaxdense_noise/add:z:0*
T0*'
_output_shapes
:���������
2
dense_noise/Softmax�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_15_minimum_readvariableop_resource;^activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_16_minimum_readvariableop_resource;^activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_17_minimum_readvariableop_resource;^activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_18_minimum_readvariableop_resource;^activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentitydense_noise/Softmax:softmax:0;^activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_15/Minimum/ReadVariableOp<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp;^activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_16/Minimum/ReadVariableOp<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp;^activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_17/Minimum/ReadVariableOp<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp;^activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_18/Minimum/ReadVariableOp<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp7^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_1'^conv2d_noise_18/BiasAdd/ReadVariableOp+^conv2d_noise_18/convolution/ReadVariableOp'^conv2d_noise_19/BiasAdd/ReadVariableOp+^conv2d_noise_19/convolution/ReadVariableOp'^conv2d_noise_20/BiasAdd/ReadVariableOp+^conv2d_noise_20/convolution/ReadVariableOp"^dense_noise/MatMul/ReadVariableOp^dense_noise/add/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@::::::::::::::::::::::::2x
:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_15/Minimum/ReadVariableOp*activation_quant_15/Minimum/ReadVariableOp2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_16/Minimum/ReadVariableOp*activation_quant_16/Minimum/ReadVariableOp2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_17/Minimum/ReadVariableOp*activation_quant_17/Minimum/ReadVariableOp2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_18/Minimum/ReadVariableOp*activation_quant_18/Minimum/ReadVariableOp2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp2p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12P
&conv2d_noise_18/BiasAdd/ReadVariableOp&conv2d_noise_18/BiasAdd/ReadVariableOp2X
*conv2d_noise_18/convolution/ReadVariableOp*conv2d_noise_18/convolution/ReadVariableOp2P
&conv2d_noise_19/BiasAdd/ReadVariableOp&conv2d_noise_19/BiasAdd/ReadVariableOp2X
*conv2d_noise_19/convolution/ReadVariableOp*conv2d_noise_19/convolution/ReadVariableOp2P
&conv2d_noise_20/BiasAdd/ReadVariableOp&conv2d_noise_20/BiasAdd/ReadVariableOp2X
*conv2d_noise_20/convolution/ReadVariableOp*conv2d_noise_20/convolution/ReadVariableOp2F
!dense_noise/MatMul/ReadVariableOp!dense_noise/MatMul/ReadVariableOp2@
dense_noise/add/ReadVariableOpdense_noise/add/ReadVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_120388

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
i
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_118162

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
,__inference_dense_noise_layer_call_fn_120593
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*P
fKRI
G__inference_dense_noise_layer_call_and_return_conditional_losses_1187942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
k
A__inference_add_1_layer_call_and_return_conditional_losses_118698

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�	
�
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_119798
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�
P
$__inference_add_layer_call_fn_119984
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1183472
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
__inference_loss_fn_3_120645H
Dactivation_quant_18_relux_regularizer_square_readvariableop_resource
identity��;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOpDactivation_quant_18_relux_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentity-activation_quant_18/relux/Regularizer/add:z:0<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp
�
�
&__inference_model_layer_call_fn_119029
input_1
input_2"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25*%
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1190022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
�
�
7__inference_batch_normalization_17_layer_call_fn_120158

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1185002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_118317

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
0__inference_conv2d_noise_20_layer_call_fn_120313
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_1185842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_120214

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�	
�
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_118594
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�
�
,__inference_dense_noise_layer_call_fn_120586
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*P
fKRI
G__inference_dense_noise_layer_call_and_return_conditional_losses_1187832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_118584
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�	
�
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_118426
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�
�
7__inference_batch_normalization_16_layer_call_fn_119898

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1183172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_118714

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������@2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_16_layer_call_fn_119972

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1178852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
&__inference_model_layer_call_fn_119135
input_1
input_2"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25*%
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1191082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
�
�
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_118416
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
��
�
A__inference_model_layer_call_and_return_conditional_losses_118846
input_1
input_26
2activation_quant_15_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_19
5batch_normalization_16_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_39
5batch_normalization_16_statefulpartitionedcall_args_46
2activation_quant_16_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_19
5batch_normalization_17_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_39
5batch_normalization_17_statefulpartitionedcall_args_46
2activation_quant_17_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_19
5batch_normalization_18_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_39
5batch_normalization_18_statefulpartitionedcall_args_46
2activation_quant_18_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_2
identity��+activation_quant_15/StatefulPartitionedCall�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�+activation_quant_16/StatefulPartitionedCall�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�+activation_quant_17/StatefulPartitionedCall�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�+activation_quant_18/StatefulPartitionedCall�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�'conv2d_noise_18/StatefulPartitionedCall�'conv2d_noise_19/StatefulPartitionedCall�'conv2d_noise_20/StatefulPartitionedCall�#dense_noise/StatefulPartitionedCall�
+activation_quant_15/StatefulPartitionedCallStatefulPartitionedCallinput_22activation_quant_15_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_1181932-
+activation_quant_15/StatefulPartitionedCall�
'conv2d_noise_18/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_15/StatefulPartitionedCall:output:0.conv2d_noise_18_statefulpartitionedcall_args_1.conv2d_noise_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_1182332)
'conv2d_noise_18/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_18/StatefulPartitionedCall:output:05batch_normalization_16_statefulpartitionedcall_args_15batch_normalization_16_statefulpartitionedcall_args_25batch_normalization_16_statefulpartitionedcall_args_35batch_normalization_16_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_11829520
.batch_normalization_16/StatefulPartitionedCall�
add/PartitionedCallPartitionedCallinput_17batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1183472
add/PartitionedCall�
+activation_quant_16/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:02activation_quant_16_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_1183762-
+activation_quant_16/StatefulPartitionedCall�
'conv2d_noise_19/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:0.conv2d_noise_19_statefulpartitionedcall_args_1.conv2d_noise_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_1184162)
'conv2d_noise_19/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_19/StatefulPartitionedCall:output:05batch_normalization_17_statefulpartitionedcall_args_15batch_normalization_17_statefulpartitionedcall_args_25batch_normalization_17_statefulpartitionedcall_args_35batch_normalization_17_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_11847820
.batch_normalization_17/StatefulPartitionedCall�
+activation_quant_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:02activation_quant_17_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_1185442-
+activation_quant_17/StatefulPartitionedCall�
'conv2d_noise_20/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_17/StatefulPartitionedCall:output:0.conv2d_noise_20_statefulpartitionedcall_args_1.conv2d_noise_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_1185842)
'conv2d_noise_20/StatefulPartitionedCall�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_20/StatefulPartitionedCall:output:05batch_normalization_18_statefulpartitionedcall_args_15batch_normalization_18_statefulpartitionedcall_args_25batch_normalization_18_statefulpartitionedcall_args_35batch_normalization_18_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_11864620
.batch_normalization_18/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:07batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_1186982
add_1/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_1181622#
!average_pooling2d/PartitionedCall�
flatten/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1187142
flatten/PartitionedCall�
+activation_quant_18/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:02activation_quant_18_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_1187422-
+activation_quant_18/StatefulPartitionedCall�
#dense_noise/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_18/StatefulPartitionedCall:output:0*dense_noise_statefulpartitionedcall_args_1*dense_noise_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*P
fKRI
G__inference_dense_noise_layer_call_and_return_conditional_losses_1187832%
#dense_noise/StatefulPartitionedCall�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_15_statefulpartitionedcall_args_1,^activation_quant_15/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_16_statefulpartitionedcall_args_1,^activation_quant_16/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_17_statefulpartitionedcall_args_1,^activation_quant_17/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_18_statefulpartitionedcall_args_1,^activation_quant_18/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentity,dense_noise/StatefulPartitionedCall:output:0,^activation_quant_15/StatefulPartitionedCall<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp,^activation_quant_16/StatefulPartitionedCall<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp,^activation_quant_17/StatefulPartitionedCall<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp,^activation_quant_18/StatefulPartitionedCall<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall(^conv2d_noise_18/StatefulPartitionedCall(^conv2d_noise_19/StatefulPartitionedCall(^conv2d_noise_20/StatefulPartitionedCall$^dense_noise/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@::::::::::::::::::::::::2Z
+activation_quant_15/StatefulPartitionedCall+activation_quant_15/StatefulPartitionedCall2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_16/StatefulPartitionedCall+activation_quant_16/StatefulPartitionedCall2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_17/StatefulPartitionedCall+activation_quant_17/StatefulPartitionedCall2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_18/StatefulPartitionedCall+activation_quant_18/StatefulPartitionedCall2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2R
'conv2d_noise_18/StatefulPartitionedCall'conv2d_noise_18/StatefulPartitionedCall2R
'conv2d_noise_19/StatefulPartitionedCall'conv2d_noise_19/StatefulPartitionedCall2R
'conv2d_noise_20/StatefulPartitionedCall'conv2d_noise_20/StatefulPartitionedCall2J
#dense_noise/StatefulPartitionedCall#dense_noise/StatefulPartitionedCall:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
�
�
&__inference_model_layer_call_fn_119694
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25*%
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1190022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
��
�
A__inference_model_layer_call_and_return_conditional_losses_119511
inputs_0
inputs_17
3activation_quant_15_minimum_readvariableop_resource/
+conv2d_noise_18_abs_readvariableop_resource-
)conv2d_noise_18_readvariableop_1_resource2
.batch_normalization_16_readvariableop_resource4
0batch_normalization_16_readvariableop_1_resource1
-batch_normalization_16_assignmovingavg_1192963
/batch_normalization_16_assignmovingavg_1_1193037
3activation_quant_16_minimum_readvariableop_resource/
+conv2d_noise_19_abs_readvariableop_resource-
)conv2d_noise_19_readvariableop_1_resource2
.batch_normalization_17_readvariableop_resource4
0batch_normalization_17_readvariableop_1_resource1
-batch_normalization_17_assignmovingavg_1193613
/batch_normalization_17_assignmovingavg_1_1193687
3activation_quant_17_minimum_readvariableop_resource/
+conv2d_noise_20_abs_readvariableop_resource-
)conv2d_noise_20_readvariableop_1_resource2
.batch_normalization_18_readvariableop_resource4
0batch_normalization_18_readvariableop_1_resource1
-batch_normalization_18_assignmovingavg_1194253
/batch_normalization_18_assignmovingavg_1_1194327
3activation_quant_18_minimum_readvariableop_resource+
'dense_noise_abs_readvariableop_resource)
%dense_noise_readvariableop_1_resource
identity��:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_15/Minimum/ReadVariableOp�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_16/Minimum/ReadVariableOp�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_17/Minimum/ReadVariableOp�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp�*activation_quant_18/Minimum/ReadVariableOp�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_16/AssignMovingAvg/ReadVariableOp�<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�%batch_normalization_16/ReadVariableOp�'batch_normalization_16/ReadVariableOp_1�:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_17/AssignMovingAvg/ReadVariableOp�<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp�%batch_normalization_17/ReadVariableOp�'batch_normalization_17/ReadVariableOp_1�:batch_normalization_18/AssignMovingAvg/AssignSubVariableOp�5batch_normalization_18/AssignMovingAvg/ReadVariableOp�<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp�7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp�%batch_normalization_18/ReadVariableOp�'batch_normalization_18/ReadVariableOp_1�"conv2d_noise_18/Abs/ReadVariableOp�conv2d_noise_18/ReadVariableOp� conv2d_noise_18/ReadVariableOp_1�"conv2d_noise_19/Abs/ReadVariableOp�conv2d_noise_19/ReadVariableOp� conv2d_noise_19/ReadVariableOp_1�"conv2d_noise_20/Abs/ReadVariableOp�conv2d_noise_20/ReadVariableOp� conv2d_noise_20/ReadVariableOp_1�dense_noise/Abs/ReadVariableOp�dense_noise/ReadVariableOp�dense_noise/ReadVariableOp_1�
*activation_quant_15/Minimum/ReadVariableOpReadVariableOp3activation_quant_15_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_15/Minimum/ReadVariableOp�
activation_quant_15/MinimumMinimuminputs_12activation_quant_15/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_15/Minimum�
activation_quant_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_15/Maximum/y�
activation_quant_15/MaximumMaximumactivation_quant_15/Minimum:z:0&activation_quant_15/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_15/Maximum{
activation_quant_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_15/Const�
:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_15_minimum_readvariableop_resource+^activation_quant_15/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_15/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_15/Maximum:z:0"activation_quant_15/Const:output:0Bactivation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_15/FakeQuantWithMinMaxVars�
"conv2d_noise_18/Abs/ReadVariableOpReadVariableOp+conv2d_noise_18_abs_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"conv2d_noise_18/Abs/ReadVariableOp�
conv2d_noise_18/AbsAbs*conv2d_noise_18/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
conv2d_noise_18/Abs�
conv2d_noise_18/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
conv2d_noise_18/Const�
conv2d_noise_18/MaxMaxconv2d_noise_18/Abs:y:0conv2d_noise_18/Const:output:0*
T0*
_output_shapes
: 2
conv2d_noise_18/Maxs
conv2d_noise_18/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_18/mul/y�
conv2d_noise_18/mulMulconv2d_noise_18/Max:output:0conv2d_noise_18/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_18/mul�
#conv2d_noise_18/random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2%
#conv2d_noise_18/random_normal/shape�
"conv2d_noise_18/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_noise_18/random_normal/mean�
2conv2d_noise_18/random_normal/RandomStandardNormalRandomStandardNormal,conv2d_noise_18/random_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype024
2conv2d_noise_18/random_normal/RandomStandardNormal�
!conv2d_noise_18/random_normal/mulMul;conv2d_noise_18/random_normal/RandomStandardNormal:output:0conv2d_noise_18/mul:z:0*
T0*&
_output_shapes
:@@2#
!conv2d_noise_18/random_normal/mul�
conv2d_noise_18/random_normalAdd%conv2d_noise_18/random_normal/mul:z:0+conv2d_noise_18/random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
conv2d_noise_18/random_normal�
conv2d_noise_18/ReadVariableOpReadVariableOp+conv2d_noise_18_abs_readvariableop_resource#^conv2d_noise_18/Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_noise_18/ReadVariableOp�
conv2d_noise_18/addAddV2&conv2d_noise_18/ReadVariableOp:value:0!conv2d_noise_18/random_normal:z:0*
T0*&
_output_shapes
:@@2
conv2d_noise_18/addw
conv2d_noise_18/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_18/mul_1/y�
conv2d_noise_18/mul_1Mulconv2d_noise_18/Max:output:0 conv2d_noise_18/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_18/mul_1�
%conv2d_noise_18/random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2'
%conv2d_noise_18/random_normal_1/shape�
$conv2d_noise_18/random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv2d_noise_18/random_normal_1/mean�
4conv2d_noise_18/random_normal_1/RandomStandardNormalRandomStandardNormal.conv2d_noise_18/random_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype026
4conv2d_noise_18/random_normal_1/RandomStandardNormal�
#conv2d_noise_18/random_normal_1/mulMul=conv2d_noise_18/random_normal_1/RandomStandardNormal:output:0conv2d_noise_18/mul_1:z:0*
T0*
_output_shapes
:@2%
#conv2d_noise_18/random_normal_1/mul�
conv2d_noise_18/random_normal_1Add'conv2d_noise_18/random_normal_1/mul:z:0-conv2d_noise_18/random_normal_1/mean:output:0*
T0*
_output_shapes
:@2!
conv2d_noise_18/random_normal_1�
 conv2d_noise_18/ReadVariableOp_1ReadVariableOp)conv2d_noise_18_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 conv2d_noise_18/ReadVariableOp_1�
conv2d_noise_18/add_1AddV2(conv2d_noise_18/ReadVariableOp_1:value:0#conv2d_noise_18/random_normal_1:z:0*
T0*
_output_shapes
:@2
conv2d_noise_18/add_1�
conv2d_noise_18/convolutionConv2D5activation_quant_15/FakeQuantWithMinMaxVars:outputs:0conv2d_noise_18/add:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_18/convolution�
conv2d_noise_18/BiasAddBiasAdd$conv2d_noise_18/convolution:output:0conv2d_noise_18/add_1:z:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_18/BiasAdd�
#batch_normalization_16/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_16/LogicalAnd/x�
#batch_normalization_16/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_16/LogicalAnd/y�
!batch_normalization_16/LogicalAnd
LogicalAnd,batch_normalization_16/LogicalAnd/x:output:0,batch_normalization_16/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_16/LogicalAnd�
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_16/ReadVariableOp�
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_16/ReadVariableOp_1
batch_normalization_16/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_16/Const�
batch_normalization_16/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
batch_normalization_16/Const_1�
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_18/BiasAdd:output:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0%batch_normalization_16/Const:output:0'batch_normalization_16/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2)
'batch_normalization_16/FusedBatchNormV3�
batch_normalization_16/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2 
batch_normalization_16/Const_2�
,batch_normalization_16/AssignMovingAvg/sub/xConst*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/119296*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,batch_normalization_16/AssignMovingAvg/sub/x�
*batch_normalization_16/AssignMovingAvg/subSub5batch_normalization_16/AssignMovingAvg/sub/x:output:0'batch_normalization_16/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/119296*
_output_shapes
: 2,
*batch_normalization_16/AssignMovingAvg/sub�
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_16_assignmovingavg_119296*
_output_shapes
:@*
dtype027
5batch_normalization_16/AssignMovingAvg/ReadVariableOp�
,batch_normalization_16/AssignMovingAvg/sub_1Sub=batch_normalization_16/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_16/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/119296*
_output_shapes
:@2.
,batch_normalization_16/AssignMovingAvg/sub_1�
*batch_normalization_16/AssignMovingAvg/mulMul0batch_normalization_16/AssignMovingAvg/sub_1:z:0.batch_normalization_16/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/119296*
_output_shapes
:@2,
*batch_normalization_16/AssignMovingAvg/mul�
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_16_assignmovingavg_119296.batch_normalization_16/AssignMovingAvg/mul:z:06^batch_normalization_16/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_16/AssignMovingAvg/119296*
_output_shapes
 *
dtype02<
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_16/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/119303*
_output_shapes
: *
dtype0*
valueB
 *  �?20
.batch_normalization_16/AssignMovingAvg_1/sub/x�
,batch_normalization_16/AssignMovingAvg_1/subSub7batch_normalization_16/AssignMovingAvg_1/sub/x:output:0'batch_normalization_16/Const_2:output:0*
T0*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/119303*
_output_shapes
: 2.
,batch_normalization_16/AssignMovingAvg_1/sub�
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_16_assignmovingavg_1_119303*
_output_shapes
:@*
dtype029
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp�
.batch_normalization_16/AssignMovingAvg_1/sub_1Sub?batch_normalization_16/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_16/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/119303*
_output_shapes
:@20
.batch_normalization_16/AssignMovingAvg_1/sub_1�
,batch_normalization_16/AssignMovingAvg_1/mulMul2batch_normalization_16/AssignMovingAvg_1/sub_1:z:00batch_normalization_16/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/119303*
_output_shapes
:@2.
,batch_normalization_16/AssignMovingAvg_1/mul�
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_16_assignmovingavg_1_1193030batch_normalization_16/AssignMovingAvg_1/mul:z:08^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_16/AssignMovingAvg_1/119303*
_output_shapes
 *
dtype02>
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp�
add/addAddV2inputs_0+batch_normalization_16/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2	
add/add�
*activation_quant_16/Minimum/ReadVariableOpReadVariableOp3activation_quant_16_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_16/Minimum/ReadVariableOp�
activation_quant_16/MinimumMinimumadd/add:z:02activation_quant_16/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_16/Minimum�
activation_quant_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_16/Maximum/y�
activation_quant_16/MaximumMaximumactivation_quant_16/Minimum:z:0&activation_quant_16/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_16/Maximum{
activation_quant_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_16/Const�
:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_16_minimum_readvariableop_resource+^activation_quant_16/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_16/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_16/Maximum:z:0"activation_quant_16/Const:output:0Bactivation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_16/FakeQuantWithMinMaxVars�
"conv2d_noise_19/Abs/ReadVariableOpReadVariableOp+conv2d_noise_19_abs_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"conv2d_noise_19/Abs/ReadVariableOp�
conv2d_noise_19/AbsAbs*conv2d_noise_19/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
conv2d_noise_19/Abs�
conv2d_noise_19/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
conv2d_noise_19/Const�
conv2d_noise_19/MaxMaxconv2d_noise_19/Abs:y:0conv2d_noise_19/Const:output:0*
T0*
_output_shapes
: 2
conv2d_noise_19/Maxs
conv2d_noise_19/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_19/mul/y�
conv2d_noise_19/mulMulconv2d_noise_19/Max:output:0conv2d_noise_19/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_19/mul�
#conv2d_noise_19/random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2%
#conv2d_noise_19/random_normal/shape�
"conv2d_noise_19/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_noise_19/random_normal/mean�
2conv2d_noise_19/random_normal/RandomStandardNormalRandomStandardNormal,conv2d_noise_19/random_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype024
2conv2d_noise_19/random_normal/RandomStandardNormal�
!conv2d_noise_19/random_normal/mulMul;conv2d_noise_19/random_normal/RandomStandardNormal:output:0conv2d_noise_19/mul:z:0*
T0*&
_output_shapes
:@@2#
!conv2d_noise_19/random_normal/mul�
conv2d_noise_19/random_normalAdd%conv2d_noise_19/random_normal/mul:z:0+conv2d_noise_19/random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
conv2d_noise_19/random_normal�
conv2d_noise_19/ReadVariableOpReadVariableOp+conv2d_noise_19_abs_readvariableop_resource#^conv2d_noise_19/Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_noise_19/ReadVariableOp�
conv2d_noise_19/addAddV2&conv2d_noise_19/ReadVariableOp:value:0!conv2d_noise_19/random_normal:z:0*
T0*&
_output_shapes
:@@2
conv2d_noise_19/addw
conv2d_noise_19/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_19/mul_1/y�
conv2d_noise_19/mul_1Mulconv2d_noise_19/Max:output:0 conv2d_noise_19/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_19/mul_1�
%conv2d_noise_19/random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2'
%conv2d_noise_19/random_normal_1/shape�
$conv2d_noise_19/random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv2d_noise_19/random_normal_1/mean�
4conv2d_noise_19/random_normal_1/RandomStandardNormalRandomStandardNormal.conv2d_noise_19/random_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype026
4conv2d_noise_19/random_normal_1/RandomStandardNormal�
#conv2d_noise_19/random_normal_1/mulMul=conv2d_noise_19/random_normal_1/RandomStandardNormal:output:0conv2d_noise_19/mul_1:z:0*
T0*
_output_shapes
:@2%
#conv2d_noise_19/random_normal_1/mul�
conv2d_noise_19/random_normal_1Add'conv2d_noise_19/random_normal_1/mul:z:0-conv2d_noise_19/random_normal_1/mean:output:0*
T0*
_output_shapes
:@2!
conv2d_noise_19/random_normal_1�
 conv2d_noise_19/ReadVariableOp_1ReadVariableOp)conv2d_noise_19_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 conv2d_noise_19/ReadVariableOp_1�
conv2d_noise_19/add_1AddV2(conv2d_noise_19/ReadVariableOp_1:value:0#conv2d_noise_19/random_normal_1:z:0*
T0*
_output_shapes
:@2
conv2d_noise_19/add_1�
conv2d_noise_19/convolutionConv2D5activation_quant_16/FakeQuantWithMinMaxVars:outputs:0conv2d_noise_19/add:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_19/convolution�
conv2d_noise_19/BiasAddBiasAdd$conv2d_noise_19/convolution:output:0conv2d_noise_19/add_1:z:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_19/BiasAdd�
#batch_normalization_17/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_17/LogicalAnd/x�
#batch_normalization_17/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_17/LogicalAnd/y�
!batch_normalization_17/LogicalAnd
LogicalAnd,batch_normalization_17/LogicalAnd/x:output:0,batch_normalization_17/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_17/LogicalAnd�
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_17/ReadVariableOp�
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_17/ReadVariableOp_1
batch_normalization_17/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_17/Const�
batch_normalization_17/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
batch_normalization_17/Const_1�
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_19/BiasAdd:output:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0%batch_normalization_17/Const:output:0'batch_normalization_17/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2)
'batch_normalization_17/FusedBatchNormV3�
batch_normalization_17/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2 
batch_normalization_17/Const_2�
,batch_normalization_17/AssignMovingAvg/sub/xConst*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/119361*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,batch_normalization_17/AssignMovingAvg/sub/x�
*batch_normalization_17/AssignMovingAvg/subSub5batch_normalization_17/AssignMovingAvg/sub/x:output:0'batch_normalization_17/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/119361*
_output_shapes
: 2,
*batch_normalization_17/AssignMovingAvg/sub�
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_17_assignmovingavg_119361*
_output_shapes
:@*
dtype027
5batch_normalization_17/AssignMovingAvg/ReadVariableOp�
,batch_normalization_17/AssignMovingAvg/sub_1Sub=batch_normalization_17/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_17/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/119361*
_output_shapes
:@2.
,batch_normalization_17/AssignMovingAvg/sub_1�
*batch_normalization_17/AssignMovingAvg/mulMul0batch_normalization_17/AssignMovingAvg/sub_1:z:0.batch_normalization_17/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/119361*
_output_shapes
:@2,
*batch_normalization_17/AssignMovingAvg/mul�
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_17_assignmovingavg_119361.batch_normalization_17/AssignMovingAvg/mul:z:06^batch_normalization_17/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_17/AssignMovingAvg/119361*
_output_shapes
 *
dtype02<
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_17/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/119368*
_output_shapes
: *
dtype0*
valueB
 *  �?20
.batch_normalization_17/AssignMovingAvg_1/sub/x�
,batch_normalization_17/AssignMovingAvg_1/subSub7batch_normalization_17/AssignMovingAvg_1/sub/x:output:0'batch_normalization_17/Const_2:output:0*
T0*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/119368*
_output_shapes
: 2.
,batch_normalization_17/AssignMovingAvg_1/sub�
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_17_assignmovingavg_1_119368*
_output_shapes
:@*
dtype029
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp�
.batch_normalization_17/AssignMovingAvg_1/sub_1Sub?batch_normalization_17/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_17/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/119368*
_output_shapes
:@20
.batch_normalization_17/AssignMovingAvg_1/sub_1�
,batch_normalization_17/AssignMovingAvg_1/mulMul2batch_normalization_17/AssignMovingAvg_1/sub_1:z:00batch_normalization_17/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/119368*
_output_shapes
:@2.
,batch_normalization_17/AssignMovingAvg_1/mul�
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_17_assignmovingavg_1_1193680batch_normalization_17/AssignMovingAvg_1/mul:z:08^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_17/AssignMovingAvg_1/119368*
_output_shapes
 *
dtype02>
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp�
*activation_quant_17/Minimum/ReadVariableOpReadVariableOp3activation_quant_17_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_17/Minimum/ReadVariableOp�
activation_quant_17/MinimumMinimum+batch_normalization_17/FusedBatchNormV3:y:02activation_quant_17/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
activation_quant_17/Minimum�
activation_quant_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_17/Maximum/y�
activation_quant_17/MaximumMaximumactivation_quant_17/Minimum:z:0&activation_quant_17/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2
activation_quant_17/Maximum{
activation_quant_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_17/Const�
:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_17_minimum_readvariableop_resource+^activation_quant_17/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_17/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_17/Maximum:z:0"activation_quant_17/Const:output:0Bactivation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2-
+activation_quant_17/FakeQuantWithMinMaxVars�
"conv2d_noise_20/Abs/ReadVariableOpReadVariableOp+conv2d_noise_20_abs_readvariableop_resource*&
_output_shapes
:@@*
dtype02$
"conv2d_noise_20/Abs/ReadVariableOp�
conv2d_noise_20/AbsAbs*conv2d_noise_20/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
conv2d_noise_20/Abs�
conv2d_noise_20/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
conv2d_noise_20/Const�
conv2d_noise_20/MaxMaxconv2d_noise_20/Abs:y:0conv2d_noise_20/Const:output:0*
T0*
_output_shapes
: 2
conv2d_noise_20/Maxs
conv2d_noise_20/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_20/mul/y�
conv2d_noise_20/mulMulconv2d_noise_20/Max:output:0conv2d_noise_20/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_20/mul�
#conv2d_noise_20/random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2%
#conv2d_noise_20/random_normal/shape�
"conv2d_noise_20/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_noise_20/random_normal/mean�
2conv2d_noise_20/random_normal/RandomStandardNormalRandomStandardNormal,conv2d_noise_20/random_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype024
2conv2d_noise_20/random_normal/RandomStandardNormal�
!conv2d_noise_20/random_normal/mulMul;conv2d_noise_20/random_normal/RandomStandardNormal:output:0conv2d_noise_20/mul:z:0*
T0*&
_output_shapes
:@@2#
!conv2d_noise_20/random_normal/mul�
conv2d_noise_20/random_normalAdd%conv2d_noise_20/random_normal/mul:z:0+conv2d_noise_20/random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
conv2d_noise_20/random_normal�
conv2d_noise_20/ReadVariableOpReadVariableOp+conv2d_noise_20_abs_readvariableop_resource#^conv2d_noise_20/Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02 
conv2d_noise_20/ReadVariableOp�
conv2d_noise_20/addAddV2&conv2d_noise_20/ReadVariableOp:value:0!conv2d_noise_20/random_normal:z:0*
T0*&
_output_shapes
:@@2
conv2d_noise_20/addw
conv2d_noise_20/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
conv2d_noise_20/mul_1/y�
conv2d_noise_20/mul_1Mulconv2d_noise_20/Max:output:0 conv2d_noise_20/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_noise_20/mul_1�
%conv2d_noise_20/random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2'
%conv2d_noise_20/random_normal_1/shape�
$conv2d_noise_20/random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$conv2d_noise_20/random_normal_1/mean�
4conv2d_noise_20/random_normal_1/RandomStandardNormalRandomStandardNormal.conv2d_noise_20/random_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype026
4conv2d_noise_20/random_normal_1/RandomStandardNormal�
#conv2d_noise_20/random_normal_1/mulMul=conv2d_noise_20/random_normal_1/RandomStandardNormal:output:0conv2d_noise_20/mul_1:z:0*
T0*
_output_shapes
:@2%
#conv2d_noise_20/random_normal_1/mul�
conv2d_noise_20/random_normal_1Add'conv2d_noise_20/random_normal_1/mul:z:0-conv2d_noise_20/random_normal_1/mean:output:0*
T0*
_output_shapes
:@2!
conv2d_noise_20/random_normal_1�
 conv2d_noise_20/ReadVariableOp_1ReadVariableOp)conv2d_noise_20_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 conv2d_noise_20/ReadVariableOp_1�
conv2d_noise_20/add_1AddV2(conv2d_noise_20/ReadVariableOp_1:value:0#conv2d_noise_20/random_normal_1:z:0*
T0*
_output_shapes
:@2
conv2d_noise_20/add_1�
conv2d_noise_20/convolutionConv2D5activation_quant_17/FakeQuantWithMinMaxVars:outputs:0conv2d_noise_20/add:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_noise_20/convolution�
conv2d_noise_20/BiasAddBiasAdd$conv2d_noise_20/convolution:output:0conv2d_noise_20/add_1:z:0*
T0*/
_output_shapes
:���������@2
conv2d_noise_20/BiasAdd�
#batch_normalization_18/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_18/LogicalAnd/x�
#batch_normalization_18/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2%
#batch_normalization_18/LogicalAnd/y�
!batch_normalization_18/LogicalAnd
LogicalAnd,batch_normalization_18/LogicalAnd/x:output:0,batch_normalization_18/LogicalAnd/y:output:0*
_output_shapes
: 2#
!batch_normalization_18/LogicalAnd�
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes
:@*
dtype02'
%batch_normalization_18/ReadVariableOp�
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes
:@*
dtype02)
'batch_normalization_18/ReadVariableOp_1
batch_normalization_18/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_18/Const�
batch_normalization_18/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2 
batch_normalization_18/Const_1�
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3 conv2d_noise_20/BiasAdd:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0%batch_normalization_18/Const:output:0'batch_normalization_18/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2)
'batch_normalization_18/FusedBatchNormV3�
batch_normalization_18/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2 
batch_normalization_18/Const_2�
,batch_normalization_18/AssignMovingAvg/sub/xConst*@
_class6
42loc:@batch_normalization_18/AssignMovingAvg/119425*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,batch_normalization_18/AssignMovingAvg/sub/x�
*batch_normalization_18/AssignMovingAvg/subSub5batch_normalization_18/AssignMovingAvg/sub/x:output:0'batch_normalization_18/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_18/AssignMovingAvg/119425*
_output_shapes
: 2,
*batch_normalization_18/AssignMovingAvg/sub�
5batch_normalization_18/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_18_assignmovingavg_119425*
_output_shapes
:@*
dtype027
5batch_normalization_18/AssignMovingAvg/ReadVariableOp�
,batch_normalization_18/AssignMovingAvg/sub_1Sub=batch_normalization_18/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_18/FusedBatchNormV3:batch_mean:0*
T0*@
_class6
42loc:@batch_normalization_18/AssignMovingAvg/119425*
_output_shapes
:@2.
,batch_normalization_18/AssignMovingAvg/sub_1�
*batch_normalization_18/AssignMovingAvg/mulMul0batch_normalization_18/AssignMovingAvg/sub_1:z:0.batch_normalization_18/AssignMovingAvg/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_18/AssignMovingAvg/119425*
_output_shapes
:@2,
*batch_normalization_18/AssignMovingAvg/mul�
:batch_normalization_18/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_18_assignmovingavg_119425.batch_normalization_18/AssignMovingAvg/mul:z:06^batch_normalization_18/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_18/AssignMovingAvg/119425*
_output_shapes
 *
dtype02<
:batch_normalization_18/AssignMovingAvg/AssignSubVariableOp�
.batch_normalization_18/AssignMovingAvg_1/sub/xConst*B
_class8
64loc:@batch_normalization_18/AssignMovingAvg_1/119432*
_output_shapes
: *
dtype0*
valueB
 *  �?20
.batch_normalization_18/AssignMovingAvg_1/sub/x�
,batch_normalization_18/AssignMovingAvg_1/subSub7batch_normalization_18/AssignMovingAvg_1/sub/x:output:0'batch_normalization_18/Const_2:output:0*
T0*B
_class8
64loc:@batch_normalization_18/AssignMovingAvg_1/119432*
_output_shapes
: 2.
,batch_normalization_18/AssignMovingAvg_1/sub�
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_18_assignmovingavg_1_119432*
_output_shapes
:@*
dtype029
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp�
.batch_normalization_18/AssignMovingAvg_1/sub_1Sub?batch_normalization_18/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_18/FusedBatchNormV3:batch_variance:0*
T0*B
_class8
64loc:@batch_normalization_18/AssignMovingAvg_1/119432*
_output_shapes
:@20
.batch_normalization_18/AssignMovingAvg_1/sub_1�
,batch_normalization_18/AssignMovingAvg_1/mulMul2batch_normalization_18/AssignMovingAvg_1/sub_1:z:00batch_normalization_18/AssignMovingAvg_1/sub:z:0*
T0*B
_class8
64loc:@batch_normalization_18/AssignMovingAvg_1/119432*
_output_shapes
:@2.
,batch_normalization_18/AssignMovingAvg_1/mul�
<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_18_assignmovingavg_1_1194320batch_normalization_18/AssignMovingAvg_1/mul:z:08^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_18/AssignMovingAvg_1/119432*
_output_shapes
 *
dtype02>
<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp�
	add_1/addAddV25activation_quant_16/FakeQuantWithMinMaxVars:outputs:0+batch_normalization_18/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
	add_1/add�
average_pooling2d/AvgPoolAvgPooladd_1/add:z:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   2
flatten/Const�
flatten/ReshapeReshape"average_pooling2d/AvgPool:output:0flatten/Const:output:0*
T0*'
_output_shapes
:���������@2
flatten/Reshape�
*activation_quant_18/Minimum/ReadVariableOpReadVariableOp3activation_quant_18_minimum_readvariableop_resource*
_output_shapes
: *
dtype02,
*activation_quant_18/Minimum/ReadVariableOp�
activation_quant_18/MinimumMinimumflatten/Reshape:output:02activation_quant_18/Minimum/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
activation_quant_18/Minimum�
activation_quant_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_18/Maximum/y�
activation_quant_18/MaximumMaximumactivation_quant_18/Minimum:z:0&activation_quant_18/Maximum/y:output:0*
T0*'
_output_shapes
:���������@2
activation_quant_18/Maximum{
activation_quant_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
activation_quant_18/Const�
:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp3activation_quant_18_minimum_readvariableop_resource+^activation_quant_18/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02<
:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp�
+activation_quant_18/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsactivation_quant_18/Maximum:z:0"activation_quant_18/Const:output:0Bactivation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*'
_output_shapes
:���������@*
num_bits2-
+activation_quant_18/FakeQuantWithMinMaxVars�
dense_noise/Abs/ReadVariableOpReadVariableOp'dense_noise_abs_readvariableop_resource*
_output_shapes

:@
*
dtype02 
dense_noise/Abs/ReadVariableOpz
dense_noise/AbsAbs&dense_noise/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@
2
dense_noise/Absw
dense_noise/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_noise/Const{
dense_noise/MaxMaxdense_noise/Abs:y:0dense_noise/Const:output:0*
T0*
_output_shapes
: 2
dense_noise/Maxk
dense_noise/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dense_noise/mul/y�
dense_noise/mulMuldense_noise/Max:output:0dense_noise/mul/y:output:0*
T0*
_output_shapes
: 2
dense_noise/mul�
dense_noise/random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   
   2!
dense_noise/random_normal/shape�
dense_noise/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense_noise/random_normal/mean�
.dense_noise/random_normal/RandomStandardNormalRandomStandardNormal(dense_noise/random_normal/shape:output:0*
T0*
_output_shapes

:@
*
dtype020
.dense_noise/random_normal/RandomStandardNormal�
dense_noise/random_normal/mulMul7dense_noise/random_normal/RandomStandardNormal:output:0dense_noise/mul:z:0*
T0*
_output_shapes

:@
2
dense_noise/random_normal/mul�
dense_noise/random_normalAdd!dense_noise/random_normal/mul:z:0'dense_noise/random_normal/mean:output:0*
T0*
_output_shapes

:@
2
dense_noise/random_normal�
dense_noise/ReadVariableOpReadVariableOp'dense_noise_abs_readvariableop_resource^dense_noise/Abs/ReadVariableOp*
_output_shapes

:@
*
dtype02
dense_noise/ReadVariableOp�
dense_noise/addAddV2"dense_noise/ReadVariableOp:value:0dense_noise/random_normal:z:0*
T0*
_output_shapes

:@
2
dense_noise/addo
dense_noise/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dense_noise/mul_1/y�
dense_noise/mul_1Muldense_noise/Max:output:0dense_noise/mul_1/y:output:0*
T0*
_output_shapes
: 2
dense_noise/mul_1�
!dense_noise/random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
2#
!dense_noise/random_normal_1/shape�
 dense_noise/random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_noise/random_normal_1/mean�
0dense_noise/random_normal_1/RandomStandardNormalRandomStandardNormal*dense_noise/random_normal_1/shape:output:0*
T0*
_output_shapes
:
*
dtype022
0dense_noise/random_normal_1/RandomStandardNormal�
dense_noise/random_normal_1/mulMul9dense_noise/random_normal_1/RandomStandardNormal:output:0dense_noise/mul_1:z:0*
T0*
_output_shapes
:
2!
dense_noise/random_normal_1/mul�
dense_noise/random_normal_1Add#dense_noise/random_normal_1/mul:z:0)dense_noise/random_normal_1/mean:output:0*
T0*
_output_shapes
:
2
dense_noise/random_normal_1�
dense_noise/ReadVariableOp_1ReadVariableOp%dense_noise_readvariableop_1_resource*
_output_shapes
:
*
dtype02
dense_noise/ReadVariableOp_1�
dense_noise/add_1AddV2$dense_noise/ReadVariableOp_1:value:0dense_noise/random_normal_1:z:0*
T0*
_output_shapes
:
2
dense_noise/add_1�
dense_noise/MatMulMatMul5activation_quant_18/FakeQuantWithMinMaxVars:outputs:0dense_noise/add:z:0*
T0*'
_output_shapes
:���������
2
dense_noise/MatMul�
dense_noise/add_2AddV2dense_noise/MatMul:product:0dense_noise/add_1:z:0*
T0*'
_output_shapes
:���������
2
dense_noise/add_2~
dense_noise/SoftmaxSoftmaxdense_noise/add_2:z:0*
T0*'
_output_shapes
:���������
2
dense_noise/Softmax�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_15_minimum_readvariableop_resource;^activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_16_minimum_readvariableop_resource;^activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_17_minimum_readvariableop_resource;^activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOp3activation_quant_18_minimum_readvariableop_resource;^activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentitydense_noise/Softmax:softmax:0;^activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_15/Minimum/ReadVariableOp<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp;^activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_16/Minimum/ReadVariableOp<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp;^activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_17/Minimum/ReadVariableOp<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp;^activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp+^activation_quant_18/Minimum/ReadVariableOp<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp;^batch_normalization_16/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_16/AssignMovingAvg/ReadVariableOp=^batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_16/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1;^batch_normalization_17/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_17/AssignMovingAvg/ReadVariableOp=^batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_17/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1;^batch_normalization_18/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_18/AssignMovingAvg/ReadVariableOp=^batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_18/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_1#^conv2d_noise_18/Abs/ReadVariableOp^conv2d_noise_18/ReadVariableOp!^conv2d_noise_18/ReadVariableOp_1#^conv2d_noise_19/Abs/ReadVariableOp^conv2d_noise_19/ReadVariableOp!^conv2d_noise_19/ReadVariableOp_1#^conv2d_noise_20/Abs/ReadVariableOp^conv2d_noise_20/ReadVariableOp!^conv2d_noise_20/ReadVariableOp_1^dense_noise/Abs/ReadVariableOp^dense_noise/ReadVariableOp^dense_noise/ReadVariableOp_1*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@::::::::::::::::::::::::2x
:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_15/Minimum/ReadVariableOp*activation_quant_15/Minimum/ReadVariableOp2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_16/Minimum/ReadVariableOp*activation_quant_16/Minimum/ReadVariableOp2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_17/Minimum/ReadVariableOp*activation_quant_17/Minimum/ReadVariableOp2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp2x
:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp:activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp2X
*activation_quant_18/Minimum/ReadVariableOp*activation_quant_18/Minimum/ReadVariableOp2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp2x
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp:batch_normalization_16/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_16/AssignMovingAvg/ReadVariableOp5batch_normalization_16/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12x
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp:batch_normalization_17/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_17/AssignMovingAvg/ReadVariableOp5batch_normalization_17/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12x
:batch_normalization_18/AssignMovingAvg/AssignSubVariableOp:batch_normalization_18/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_18/AssignMovingAvg/ReadVariableOp5batch_normalization_18/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp2N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12H
"conv2d_noise_18/Abs/ReadVariableOp"conv2d_noise_18/Abs/ReadVariableOp2@
conv2d_noise_18/ReadVariableOpconv2d_noise_18/ReadVariableOp2D
 conv2d_noise_18/ReadVariableOp_1 conv2d_noise_18/ReadVariableOp_12H
"conv2d_noise_19/Abs/ReadVariableOp"conv2d_noise_19/Abs/ReadVariableOp2@
conv2d_noise_19/ReadVariableOpconv2d_noise_19/ReadVariableOp2D
 conv2d_noise_19/ReadVariableOp_1 conv2d_noise_19/ReadVariableOp_12H
"conv2d_noise_20/Abs/ReadVariableOp"conv2d_noise_20/Abs/ReadVariableOp2@
conv2d_noise_20/ReadVariableOpconv2d_noise_20/ReadVariableOp2D
 conv2d_noise_20/ReadVariableOp_1 conv2d_noise_20/ReadVariableOp_12@
dense_noise/Abs/ReadVariableOpdense_noise/Abs/ReadVariableOp28
dense_noise/ReadVariableOpdense_noise/ReadVariableOp2<
dense_noise/ReadVariableOp_1dense_noise/ReadVariableOp_1:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_119788
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_118668

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_119858

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_119843
assignmovingavg_1_119850
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/119843*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/119843*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_119843*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/119843*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/119843*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_119843AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/119843*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/119850*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/119850*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_119850*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/119850*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/119850*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_119850AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/119850*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
R
&__inference_add_1_layer_call_fn_120492
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_1186982
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
4__inference_activation_quant_15_layer_call_fn_119758
x"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_1181932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_119880

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
i
?__inference_add_layer_call_and_return_conditional_losses_118347

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
m
A__inference_add_1_layer_call_and_return_conditional_losses_120486
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_118544
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
�
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_118193
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
�
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_118376
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
�
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_120296
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
�
G__inference_dense_noise_layer_call_and_return_conditional_losses_118783
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*
_output_shapes

:@
*
dtype02
Abs/ReadVariableOpV
AbsAbsAbs/ReadVariableOp:value:0*
T0*
_output_shapes

:@
2
Abs_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul{
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   
   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*
_output_shapes

:@
*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*
_output_shapes

:@
2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes

:@
2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*
_output_shapes

:@
*
dtype02
ReadVariableOpg
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*
_output_shapes

:@
2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:
*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:
2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:
2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:
*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:
2
add_1X
MatMulMatMulxadd:z:0*
T0*'
_output_shapes
:���������
2
MatMulf
add_2AddV2MatMul:product:0	add_1:z:0*
T0*'
_output_shapes
:���������
2
add_2Z
SoftmaxSoftmax	add_2:z:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
�
4__inference_activation_quant_18_layer_call_fn_120537
x"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_1187422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0**
_input_shapes
:���������@:22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�	
�
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_120058
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
��
�
!__inference__wrapped_model_117760
input_1
input_2=
9model_activation_quant_15_minimum_readvariableop_resource=
9model_conv2d_noise_18_convolution_readvariableop_resource9
5model_conv2d_noise_18_biasadd_readvariableop_resource8
4model_batch_normalization_16_readvariableop_resource:
6model_batch_normalization_16_readvariableop_1_resourceI
Emodel_batch_normalization_16_fusedbatchnormv3_readvariableop_resourceK
Gmodel_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource=
9model_activation_quant_16_minimum_readvariableop_resource=
9model_conv2d_noise_19_convolution_readvariableop_resource9
5model_conv2d_noise_19_biasadd_readvariableop_resource8
4model_batch_normalization_17_readvariableop_resource:
6model_batch_normalization_17_readvariableop_1_resourceI
Emodel_batch_normalization_17_fusedbatchnormv3_readvariableop_resourceK
Gmodel_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource=
9model_activation_quant_17_minimum_readvariableop_resource=
9model_conv2d_noise_20_convolution_readvariableop_resource9
5model_conv2d_noise_20_biasadd_readvariableop_resource8
4model_batch_normalization_18_readvariableop_resource:
6model_batch_normalization_18_readvariableop_1_resourceI
Emodel_batch_normalization_18_fusedbatchnormv3_readvariableop_resourceK
Gmodel_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource=
9model_activation_quant_18_minimum_readvariableop_resource4
0model_dense_noise_matmul_readvariableop_resource1
-model_dense_noise_add_readvariableop_resource
identity��@model/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp�0model/activation_quant_15/Minimum/ReadVariableOp�@model/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp�0model/activation_quant_16/Minimum/ReadVariableOp�@model/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp�0model/activation_quant_17/Minimum/ReadVariableOp�@model/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp�0model/activation_quant_18/Minimum/ReadVariableOp�<model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�>model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�+model/batch_normalization_16/ReadVariableOp�-model/batch_normalization_16/ReadVariableOp_1�<model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�>model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�+model/batch_normalization_17/ReadVariableOp�-model/batch_normalization_17/ReadVariableOp_1�<model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�>model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�+model/batch_normalization_18/ReadVariableOp�-model/batch_normalization_18/ReadVariableOp_1�,model/conv2d_noise_18/BiasAdd/ReadVariableOp�0model/conv2d_noise_18/convolution/ReadVariableOp�,model/conv2d_noise_19/BiasAdd/ReadVariableOp�0model/conv2d_noise_19/convolution/ReadVariableOp�,model/conv2d_noise_20/BiasAdd/ReadVariableOp�0model/conv2d_noise_20/convolution/ReadVariableOp�'model/dense_noise/MatMul/ReadVariableOp�$model/dense_noise/add/ReadVariableOp�
0model/activation_quant_15/Minimum/ReadVariableOpReadVariableOp9model_activation_quant_15_minimum_readvariableop_resource*
_output_shapes
: *
dtype022
0model/activation_quant_15/Minimum/ReadVariableOp�
!model/activation_quant_15/MinimumMinimuminput_28model/activation_quant_15/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_15/Minimum�
#model/activation_quant_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#model/activation_quant_15/Maximum/y�
!model/activation_quant_15/MaximumMaximum%model/activation_quant_15/Minimum:z:0,model/activation_quant_15/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_15/Maximum�
model/activation_quant_15/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
model/activation_quant_15/Const�
@model/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9model_activation_quant_15_minimum_readvariableop_resource1^model/activation_quant_15/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02B
@model/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp�
1model/activation_quant_15/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%model/activation_quant_15/Maximum:z:0(model/activation_quant_15/Const:output:0Hmodel/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits23
1model/activation_quant_15/FakeQuantWithMinMaxVars�
0model/conv2d_noise_18/convolution/ReadVariableOpReadVariableOp9model_conv2d_noise_18_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/conv2d_noise_18/convolution/ReadVariableOp�
!model/conv2d_noise_18/convolutionConv2D;model/activation_quant_15/FakeQuantWithMinMaxVars:outputs:08model/conv2d_noise_18/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2#
!model/conv2d_noise_18/convolution�
,model/conv2d_noise_18/BiasAdd/ReadVariableOpReadVariableOp5model_conv2d_noise_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model/conv2d_noise_18/BiasAdd/ReadVariableOp�
model/conv2d_noise_18/BiasAddBiasAdd*model/conv2d_noise_18/convolution:output:04model/conv2d_noise_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
model/conv2d_noise_18/BiasAdd�
)model/batch_normalization_16/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)model/batch_normalization_16/LogicalAnd/x�
)model/batch_normalization_16/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2+
)model/batch_normalization_16/LogicalAnd/y�
'model/batch_normalization_16/LogicalAnd
LogicalAnd2model/batch_normalization_16/LogicalAnd/x:output:02model/batch_normalization_16/LogicalAnd/y:output:0*
_output_shapes
: 2)
'model/batch_normalization_16/LogicalAnd�
+model/batch_normalization_16/ReadVariableOpReadVariableOp4model_batch_normalization_16_readvariableop_resource*
_output_shapes
:@*
dtype02-
+model/batch_normalization_16/ReadVariableOp�
-model/batch_normalization_16/ReadVariableOp_1ReadVariableOp6model_batch_normalization_16_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-model/batch_normalization_16/ReadVariableOp_1�
<model/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpEmodel_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp�
>model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGmodel_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1�
-model/batch_normalization_16/FusedBatchNormV3FusedBatchNormV3&model/conv2d_noise_18/BiasAdd:output:03model/batch_normalization_16/ReadVariableOp:value:05model/batch_normalization_16/ReadVariableOp_1:value:0Dmodel/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Fmodel/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2/
-model/batch_normalization_16/FusedBatchNormV3�
"model/batch_normalization_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2$
"model/batch_normalization_16/Const�
model/add/addAddV2input_11model/batch_normalization_16/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
model/add/add�
0model/activation_quant_16/Minimum/ReadVariableOpReadVariableOp9model_activation_quant_16_minimum_readvariableop_resource*
_output_shapes
: *
dtype022
0model/activation_quant_16/Minimum/ReadVariableOp�
!model/activation_quant_16/MinimumMinimummodel/add/add:z:08model/activation_quant_16/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_16/Minimum�
#model/activation_quant_16/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#model/activation_quant_16/Maximum/y�
!model/activation_quant_16/MaximumMaximum%model/activation_quant_16/Minimum:z:0,model/activation_quant_16/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_16/Maximum�
model/activation_quant_16/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
model/activation_quant_16/Const�
@model/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9model_activation_quant_16_minimum_readvariableop_resource1^model/activation_quant_16/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02B
@model/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp�
1model/activation_quant_16/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%model/activation_quant_16/Maximum:z:0(model/activation_quant_16/Const:output:0Hmodel/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits23
1model/activation_quant_16/FakeQuantWithMinMaxVars�
0model/conv2d_noise_19/convolution/ReadVariableOpReadVariableOp9model_conv2d_noise_19_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/conv2d_noise_19/convolution/ReadVariableOp�
!model/conv2d_noise_19/convolutionConv2D;model/activation_quant_16/FakeQuantWithMinMaxVars:outputs:08model/conv2d_noise_19/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2#
!model/conv2d_noise_19/convolution�
,model/conv2d_noise_19/BiasAdd/ReadVariableOpReadVariableOp5model_conv2d_noise_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model/conv2d_noise_19/BiasAdd/ReadVariableOp�
model/conv2d_noise_19/BiasAddBiasAdd*model/conv2d_noise_19/convolution:output:04model/conv2d_noise_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
model/conv2d_noise_19/BiasAdd�
)model/batch_normalization_17/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)model/batch_normalization_17/LogicalAnd/x�
)model/batch_normalization_17/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2+
)model/batch_normalization_17/LogicalAnd/y�
'model/batch_normalization_17/LogicalAnd
LogicalAnd2model/batch_normalization_17/LogicalAnd/x:output:02model/batch_normalization_17/LogicalAnd/y:output:0*
_output_shapes
: 2)
'model/batch_normalization_17/LogicalAnd�
+model/batch_normalization_17/ReadVariableOpReadVariableOp4model_batch_normalization_17_readvariableop_resource*
_output_shapes
:@*
dtype02-
+model/batch_normalization_17/ReadVariableOp�
-model/batch_normalization_17/ReadVariableOp_1ReadVariableOp6model_batch_normalization_17_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-model/batch_normalization_17/ReadVariableOp_1�
<model/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpEmodel_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp�
>model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGmodel_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1�
-model/batch_normalization_17/FusedBatchNormV3FusedBatchNormV3&model/conv2d_noise_19/BiasAdd:output:03model/batch_normalization_17/ReadVariableOp:value:05model/batch_normalization_17/ReadVariableOp_1:value:0Dmodel/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Fmodel/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2/
-model/batch_normalization_17/FusedBatchNormV3�
"model/batch_normalization_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2$
"model/batch_normalization_17/Const�
0model/activation_quant_17/Minimum/ReadVariableOpReadVariableOp9model_activation_quant_17_minimum_readvariableop_resource*
_output_shapes
: *
dtype022
0model/activation_quant_17/Minimum/ReadVariableOp�
!model/activation_quant_17/MinimumMinimum1model/batch_normalization_17/FusedBatchNormV3:y:08model/activation_quant_17/Minimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_17/Minimum�
#model/activation_quant_17/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#model/activation_quant_17/Maximum/y�
!model/activation_quant_17/MaximumMaximum%model/activation_quant_17/Minimum:z:0,model/activation_quant_17/Maximum/y:output:0*
T0*/
_output_shapes
:���������@2#
!model/activation_quant_17/Maximum�
model/activation_quant_17/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
model/activation_quant_17/Const�
@model/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9model_activation_quant_17_minimum_readvariableop_resource1^model/activation_quant_17/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02B
@model/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp�
1model/activation_quant_17/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%model/activation_quant_17/Maximum:z:0(model/activation_quant_17/Const:output:0Hmodel/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits23
1model/activation_quant_17/FakeQuantWithMinMaxVars�
0model/conv2d_noise_20/convolution/ReadVariableOpReadVariableOp9model_conv2d_noise_20_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype022
0model/conv2d_noise_20/convolution/ReadVariableOp�
!model/conv2d_noise_20/convolutionConv2D;model/activation_quant_17/FakeQuantWithMinMaxVars:outputs:08model/conv2d_noise_20/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2#
!model/conv2d_noise_20/convolution�
,model/conv2d_noise_20/BiasAdd/ReadVariableOpReadVariableOp5model_conv2d_noise_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,model/conv2d_noise_20/BiasAdd/ReadVariableOp�
model/conv2d_noise_20/BiasAddBiasAdd*model/conv2d_noise_20/convolution:output:04model/conv2d_noise_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
model/conv2d_noise_20/BiasAdd�
)model/batch_normalization_18/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)model/batch_normalization_18/LogicalAnd/x�
)model/batch_normalization_18/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2+
)model/batch_normalization_18/LogicalAnd/y�
'model/batch_normalization_18/LogicalAnd
LogicalAnd2model/batch_normalization_18/LogicalAnd/x:output:02model/batch_normalization_18/LogicalAnd/y:output:0*
_output_shapes
: 2)
'model/batch_normalization_18/LogicalAnd�
+model/batch_normalization_18/ReadVariableOpReadVariableOp4model_batch_normalization_18_readvariableop_resource*
_output_shapes
:@*
dtype02-
+model/batch_normalization_18/ReadVariableOp�
-model/batch_normalization_18/ReadVariableOp_1ReadVariableOp6model_batch_normalization_18_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-model/batch_normalization_18/ReadVariableOp_1�
<model/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpEmodel_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02>
<model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp�
>model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGmodel_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02@
>model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1�
-model/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3&model/conv2d_noise_20/BiasAdd:output:03model/batch_normalization_18/ReadVariableOp:value:05model/batch_normalization_18/ReadVariableOp_1:value:0Dmodel/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Fmodel/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2/
-model/batch_normalization_18/FusedBatchNormV3�
"model/batch_normalization_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2$
"model/batch_normalization_18/Const�
model/add_1/addAddV2;model/activation_quant_16/FakeQuantWithMinMaxVars:outputs:01model/batch_normalization_18/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������@2
model/add_1/add�
model/average_pooling2d/AvgPoolAvgPoolmodel/add_1/add:z:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2!
model/average_pooling2d/AvgPool{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   2
model/flatten/Const�
model/flatten/ReshapeReshape(model/average_pooling2d/AvgPool:output:0model/flatten/Const:output:0*
T0*'
_output_shapes
:���������@2
model/flatten/Reshape�
0model/activation_quant_18/Minimum/ReadVariableOpReadVariableOp9model_activation_quant_18_minimum_readvariableop_resource*
_output_shapes
: *
dtype022
0model/activation_quant_18/Minimum/ReadVariableOp�
!model/activation_quant_18/MinimumMinimummodel/flatten/Reshape:output:08model/activation_quant_18/Minimum/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2#
!model/activation_quant_18/Minimum�
#model/activation_quant_18/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#model/activation_quant_18/Maximum/y�
!model/activation_quant_18/MaximumMaximum%model/activation_quant_18/Minimum:z:0,model/activation_quant_18/Maximum/y:output:0*
T0*'
_output_shapes
:���������@2#
!model/activation_quant_18/Maximum�
model/activation_quant_18/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
model/activation_quant_18/Const�
@model/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp9model_activation_quant_18_minimum_readvariableop_resource1^model/activation_quant_18/Minimum/ReadVariableOp*
_output_shapes
: *
dtype02B
@model/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp�
1model/activation_quant_18/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars%model/activation_quant_18/Maximum:z:0(model/activation_quant_18/Const:output:0Hmodel/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp:value:0*'
_output_shapes
:���������@*
num_bits23
1model/activation_quant_18/FakeQuantWithMinMaxVars�
'model/dense_noise/MatMul/ReadVariableOpReadVariableOp0model_dense_noise_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02)
'model/dense_noise/MatMul/ReadVariableOp�
model/dense_noise/MatMulMatMul;model/activation_quant_18/FakeQuantWithMinMaxVars:outputs:0/model/dense_noise/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
model/dense_noise/MatMul�
$model/dense_noise/add/ReadVariableOpReadVariableOp-model_dense_noise_add_readvariableop_resource*
_output_shapes
:
*
dtype02&
$model/dense_noise/add/ReadVariableOp�
model/dense_noise/addAddV2"model/dense_noise/MatMul:product:0,model/dense_noise/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
model/dense_noise/add�
model/dense_noise/SoftmaxSoftmaxmodel/dense_noise/add:z:0*
T0*'
_output_shapes
:���������
2
model/dense_noise/Softmax�
IdentityIdentity#model/dense_noise/Softmax:softmax:0A^model/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp1^model/activation_quant_15/Minimum/ReadVariableOpA^model/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp1^model/activation_quant_16/Minimum/ReadVariableOpA^model/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp1^model/activation_quant_17/Minimum/ReadVariableOpA^model/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp1^model/activation_quant_18/Minimum/ReadVariableOp=^model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp?^model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1,^model/batch_normalization_16/ReadVariableOp.^model/batch_normalization_16/ReadVariableOp_1=^model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?^model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1,^model/batch_normalization_17/ReadVariableOp.^model/batch_normalization_17/ReadVariableOp_1=^model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp?^model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1,^model/batch_normalization_18/ReadVariableOp.^model/batch_normalization_18/ReadVariableOp_1-^model/conv2d_noise_18/BiasAdd/ReadVariableOp1^model/conv2d_noise_18/convolution/ReadVariableOp-^model/conv2d_noise_19/BiasAdd/ReadVariableOp1^model/conv2d_noise_19/convolution/ReadVariableOp-^model/conv2d_noise_20/BiasAdd/ReadVariableOp1^model/conv2d_noise_20/convolution/ReadVariableOp(^model/dense_noise/MatMul/ReadVariableOp%^model/dense_noise/add/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@::::::::::::::::::::::::2�
@model/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp@model/activation_quant_15/FakeQuantWithMinMaxVars/ReadVariableOp2d
0model/activation_quant_15/Minimum/ReadVariableOp0model/activation_quant_15/Minimum/ReadVariableOp2�
@model/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp@model/activation_quant_16/FakeQuantWithMinMaxVars/ReadVariableOp2d
0model/activation_quant_16/Minimum/ReadVariableOp0model/activation_quant_16/Minimum/ReadVariableOp2�
@model/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp@model/activation_quant_17/FakeQuantWithMinMaxVars/ReadVariableOp2d
0model/activation_quant_17/Minimum/ReadVariableOp0model/activation_quant_17/Minimum/ReadVariableOp2�
@model/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp@model/activation_quant_18/FakeQuantWithMinMaxVars/ReadVariableOp2d
0model/activation_quant_18/Minimum/ReadVariableOp0model/activation_quant_18/Minimum/ReadVariableOp2|
<model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp<model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2�
>model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1>model/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12Z
+model/batch_normalization_16/ReadVariableOp+model/batch_normalization_16/ReadVariableOp2^
-model/batch_normalization_16/ReadVariableOp_1-model/batch_normalization_16/ReadVariableOp_12|
<model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp<model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2�
>model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1>model/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12Z
+model/batch_normalization_17/ReadVariableOp+model/batch_normalization_17/ReadVariableOp2^
-model/batch_normalization_17/ReadVariableOp_1-model/batch_normalization_17/ReadVariableOp_12|
<model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp<model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2�
>model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1>model/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12Z
+model/batch_normalization_18/ReadVariableOp+model/batch_normalization_18/ReadVariableOp2^
-model/batch_normalization_18/ReadVariableOp_1-model/batch_normalization_18/ReadVariableOp_12\
,model/conv2d_noise_18/BiasAdd/ReadVariableOp,model/conv2d_noise_18/BiasAdd/ReadVariableOp2d
0model/conv2d_noise_18/convolution/ReadVariableOp0model/conv2d_noise_18/convolution/ReadVariableOp2\
,model/conv2d_noise_19/BiasAdd/ReadVariableOp,model/conv2d_noise_19/BiasAdd/ReadVariableOp2d
0model/conv2d_noise_19/convolution/ReadVariableOp0model/conv2d_noise_19/convolution/ReadVariableOp2\
,model/conv2d_noise_20/BiasAdd/ReadVariableOp,model/conv2d_noise_20/BiasAdd/ReadVariableOp2d
0model/conv2d_noise_20/convolution/ReadVariableOp0model/conv2d_noise_20/convolution/ReadVariableOp2R
'model/dense_noise/MatMul/ReadVariableOp'model/dense_noise/MatMul/ReadVariableOp2L
$model/dense_noise/add/ReadVariableOp$model/dense_noise/add/ReadVariableOp:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
�
�
$__inference_signature_wrapper_119242
input_1
input_2"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25*%
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8**
f%R#
!__inference__wrapped_model_1177602
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
�
�
0__inference_conv2d_noise_19_layer_call_fn_120072
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_1184262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�	
�
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_118243
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�
�
4__inference_activation_quant_16_layer_call_fn_120018
x"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_1183762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
4__inference_activation_quant_17_layer_call_fn_120266
x"
statefulpartitionedcall_args_1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_1185442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
7__inference_batch_normalization_16_layer_call_fn_119889

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_1182952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_120619H
Dactivation_quant_16_relux_regularizer_square_readvariableop_resource
identity��;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOpDactivation_quant_16_relux_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
IdentityIdentity-activation_quant_16/relux/Regularizer/add:z:0<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp
�	
�
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_120306
x'
#convolution_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�convolution/ReadVariableOp�
convolution/ReadVariableOpReadVariableOp#convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype02
convolution/ReadVariableOp�
convolutionConv2Dx"convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolution�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconvolution:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^convolution/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp28
convolution/ReadVariableOpconvolution/ReadVariableOp:! 

_user_specified_namex
�
k
?__inference_add_layer_call_and_return_conditional_losses_119978
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�	
�
G__inference_dense_noise_layer_call_and_return_conditional_losses_118794
x"
matmul_readvariableop_resource
add_readvariableop_resource
identity��MatMul/ReadVariableOp�add/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOpn
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:
*
dtype02
add/ReadVariableOps
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
addX
SoftmaxSoftmaxadd:z:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:! 

_user_specified_namex
�
�
7__inference_batch_normalization_17_layer_call_fn_120223

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1179862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_18_layer_call_fn_120406

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1181492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_117986

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_117971
assignmovingavg_1_117978
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/117971*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/117971*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_117971*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/117971*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/117971*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_117971AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/117971*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/117978*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/117978*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_117978*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/117978*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/117978*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_117978AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/117978*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_18_layer_call_fn_120471

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1186462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_118017

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_120012
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�
�
7__inference_batch_normalization_18_layer_call_fn_120480

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_1186682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_120531
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpr
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yp
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*'
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*'
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0**
_input_shapes
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
��
�
A__inference_model_layer_call_and_return_conditional_losses_119108

inputs
inputs_16
2activation_quant_15_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_12
.conv2d_noise_18_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_19
5batch_normalization_16_statefulpartitionedcall_args_29
5batch_normalization_16_statefulpartitionedcall_args_39
5batch_normalization_16_statefulpartitionedcall_args_46
2activation_quant_16_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_12
.conv2d_noise_19_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_19
5batch_normalization_17_statefulpartitionedcall_args_29
5batch_normalization_17_statefulpartitionedcall_args_39
5batch_normalization_17_statefulpartitionedcall_args_46
2activation_quant_17_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_12
.conv2d_noise_20_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_19
5batch_normalization_18_statefulpartitionedcall_args_29
5batch_normalization_18_statefulpartitionedcall_args_39
5batch_normalization_18_statefulpartitionedcall_args_46
2activation_quant_18_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_1.
*dense_noise_statefulpartitionedcall_args_2
identity��+activation_quant_15/StatefulPartitionedCall�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�+activation_quant_16/StatefulPartitionedCall�;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�+activation_quant_17/StatefulPartitionedCall�;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�+activation_quant_18/StatefulPartitionedCall�;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�.batch_normalization_16/StatefulPartitionedCall�.batch_normalization_17/StatefulPartitionedCall�.batch_normalization_18/StatefulPartitionedCall�'conv2d_noise_18/StatefulPartitionedCall�'conv2d_noise_19/StatefulPartitionedCall�'conv2d_noise_20/StatefulPartitionedCall�#dense_noise/StatefulPartitionedCall�
+activation_quant_15/StatefulPartitionedCallStatefulPartitionedCallinputs_12activation_quant_15_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_1181932-
+activation_quant_15/StatefulPartitionedCall�
'conv2d_noise_18/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_15/StatefulPartitionedCall:output:0.conv2d_noise_18_statefulpartitionedcall_args_1.conv2d_noise_18_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_1182432)
'conv2d_noise_18/StatefulPartitionedCall�
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_18/StatefulPartitionedCall:output:05batch_normalization_16_statefulpartitionedcall_args_15batch_normalization_16_statefulpartitionedcall_args_25batch_normalization_16_statefulpartitionedcall_args_35batch_normalization_16_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_11831720
.batch_normalization_16/StatefulPartitionedCall�
add/PartitionedCallPartitionedCallinputs7batch_normalization_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_1183472
add/PartitionedCall�
+activation_quant_16/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:02activation_quant_16_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_1183762-
+activation_quant_16/StatefulPartitionedCall�
'conv2d_noise_19/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:0.conv2d_noise_19_statefulpartitionedcall_args_1.conv2d_noise_19_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_1184262)
'conv2d_noise_19/StatefulPartitionedCall�
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_19/StatefulPartitionedCall:output:05batch_normalization_17_statefulpartitionedcall_args_15batch_normalization_17_statefulpartitionedcall_args_25batch_normalization_17_statefulpartitionedcall_args_35batch_normalization_17_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_11850020
.batch_normalization_17/StatefulPartitionedCall�
+activation_quant_17/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:02activation_quant_17_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_1185442-
+activation_quant_17/StatefulPartitionedCall�
'conv2d_noise_20/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_17/StatefulPartitionedCall:output:0.conv2d_noise_20_statefulpartitionedcall_args_1.conv2d_noise_20_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_1185942)
'conv2d_noise_20/StatefulPartitionedCall�
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall0conv2d_noise_20/StatefulPartitionedCall:output:05batch_normalization_18_statefulpartitionedcall_args_15batch_normalization_18_statefulpartitionedcall_args_25batch_normalization_18_statefulpartitionedcall_args_35batch_normalization_18_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_11866820
.batch_normalization_18/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall4activation_quant_16/StatefulPartitionedCall:output:07batch_normalization_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_1186982
add_1/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_1181622#
!average_pooling2d/PartitionedCall�
flatten/PartitionedCallPartitionedCall*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1187142
flatten/PartitionedCall�
+activation_quant_18/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:02activation_quant_18_statefulpartitionedcall_args_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*X
fSRQ
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_1187422-
+activation_quant_18/StatefulPartitionedCall�
#dense_noise/StatefulPartitionedCallStatefulPartitionedCall4activation_quant_18/StatefulPartitionedCall:output:0*dense_noise_statefulpartitionedcall_args_1*dense_noise_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*P
fKRI
G__inference_dense_noise_layer_call_and_return_conditional_losses_1187942%
#dense_noise/StatefulPartitionedCall�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_15_statefulpartitionedcall_args_1,^activation_quant_15/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
;activation_quant_16/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_16_statefulpartitionedcall_args_1,^activation_quant_16/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_16/relux/Regularizer/SquareSquareCactivation_quant_16/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_16/relux/Regularizer/Square�
+activation_quant_16/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_16/relux/Regularizer/Const�
)activation_quant_16/relux/Regularizer/SumSum0activation_quant_16/relux/Regularizer/Square:y:04activation_quant_16/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/Sum�
+activation_quant_16/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_16/relux/Regularizer/mul/x�
)activation_quant_16/relux/Regularizer/mulMul4activation_quant_16/relux/Regularizer/mul/x:output:02activation_quant_16/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/mul�
+activation_quant_16/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_16/relux/Regularizer/add/x�
)activation_quant_16/relux/Regularizer/addAddV24activation_quant_16/relux/Regularizer/add/x:output:0-activation_quant_16/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_16/relux/Regularizer/add�
;activation_quant_17/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_17_statefulpartitionedcall_args_1,^activation_quant_17/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_17/relux/Regularizer/SquareSquareCactivation_quant_17/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_17/relux/Regularizer/Square�
+activation_quant_17/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_17/relux/Regularizer/Const�
)activation_quant_17/relux/Regularizer/SumSum0activation_quant_17/relux/Regularizer/Square:y:04activation_quant_17/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/Sum�
+activation_quant_17/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_17/relux/Regularizer/mul/x�
)activation_quant_17/relux/Regularizer/mulMul4activation_quant_17/relux/Regularizer/mul/x:output:02activation_quant_17/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/mul�
+activation_quant_17/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_17/relux/Regularizer/add/x�
)activation_quant_17/relux/Regularizer/addAddV24activation_quant_17/relux/Regularizer/add/x:output:0-activation_quant_17/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_17/relux/Regularizer/add�
;activation_quant_18/relux/Regularizer/Square/ReadVariableOpReadVariableOp2activation_quant_18_statefulpartitionedcall_args_1,^activation_quant_18/StatefulPartitionedCall*
_output_shapes
: *
dtype02=
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_18/relux/Regularizer/SquareSquareCactivation_quant_18/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_18/relux/Regularizer/Square�
+activation_quant_18/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_18/relux/Regularizer/Const�
)activation_quant_18/relux/Regularizer/SumSum0activation_quant_18/relux/Regularizer/Square:y:04activation_quant_18/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/Sum�
+activation_quant_18/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_18/relux/Regularizer/mul/x�
)activation_quant_18/relux/Regularizer/mulMul4activation_quant_18/relux/Regularizer/mul/x:output:02activation_quant_18/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/mul�
+activation_quant_18/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_18/relux/Regularizer/add/x�
)activation_quant_18/relux/Regularizer/addAddV24activation_quant_18/relux/Regularizer/add/x:output:0-activation_quant_18/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_18/relux/Regularizer/add�
IdentityIdentity,dense_noise/StatefulPartitionedCall:output:0,^activation_quant_15/StatefulPartitionedCall<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp,^activation_quant_16/StatefulPartitionedCall<^activation_quant_16/relux/Regularizer/Square/ReadVariableOp,^activation_quant_17/StatefulPartitionedCall<^activation_quant_17/relux/Regularizer/Square/ReadVariableOp,^activation_quant_18/StatefulPartitionedCall<^activation_quant_18/relux/Regularizer/Square/ReadVariableOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall(^conv2d_noise_18/StatefulPartitionedCall(^conv2d_noise_19/StatefulPartitionedCall(^conv2d_noise_20/StatefulPartitionedCall$^dense_noise/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@::::::::::::::::::::::::2Z
+activation_quant_15/StatefulPartitionedCall+activation_quant_15/StatefulPartitionedCall2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_16/StatefulPartitionedCall+activation_quant_16/StatefulPartitionedCall2z
;activation_quant_16/relux/Regularizer/Square/ReadVariableOp;activation_quant_16/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_17/StatefulPartitionedCall+activation_quant_17/StatefulPartitionedCall2z
;activation_quant_17/relux/Regularizer/Square/ReadVariableOp;activation_quant_17/relux/Regularizer/Square/ReadVariableOp2Z
+activation_quant_18/StatefulPartitionedCall+activation_quant_18/StatefulPartitionedCall2z
;activation_quant_18/relux/Regularizer/Square/ReadVariableOp;activation_quant_18/relux/Regularizer/Square/ReadVariableOp2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2R
'conv2d_noise_18/StatefulPartitionedCall'conv2d_noise_18/StatefulPartitionedCall2R
'conv2d_noise_19/StatefulPartitionedCall'conv2d_noise_19/StatefulPartitionedCall2R
'conv2d_noise_20/StatefulPartitionedCall'conv2d_noise_20/StatefulPartitionedCall2J
#dense_noise/StatefulPartitionedCall#dense_noise/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
�
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_119752
x#
minimum_readvariableop_resource
identity��&FakeQuantWithMinMaxVars/ReadVariableOp�Minimum/ReadVariableOp�;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
: *
dtype02
Minimum/ReadVariableOpz
MinimumMinimumxMinimum/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
Minimum[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Maximum/yx
MaximumMaximumMinimum:z:0Maximum/y:output:0*
T0*/
_output_shapes
:���������@2	
MaximumS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const�
&FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpminimum_readvariableop_resource^Minimum/ReadVariableOp*
_output_shapes
: *
dtype02(
&FakeQuantWithMinMaxVars/ReadVariableOp�
FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMaximum:z:0Const:output:0.FakeQuantWithMinMaxVars/ReadVariableOp:value:0*/
_output_shapes
:���������@*
num_bits2
FakeQuantWithMinMaxVars�
;activation_quant_15/relux/Regularizer/Square/ReadVariableOpReadVariableOpminimum_readvariableop_resource'^FakeQuantWithMinMaxVars/ReadVariableOp*
_output_shapes
: *
dtype02=
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp�
,activation_quant_15/relux/Regularizer/SquareSquareCactivation_quant_15/relux/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2.
,activation_quant_15/relux/Regularizer/Square�
+activation_quant_15/relux/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB 2-
+activation_quant_15/relux/Regularizer/Const�
)activation_quant_15/relux/Regularizer/SumSum0activation_quant_15/relux/Regularizer/Square:y:04activation_quant_15/relux/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/Sum�
+activation_quant_15/relux/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2-
+activation_quant_15/relux/Regularizer/mul/x�
)activation_quant_15/relux/Regularizer/mulMul4activation_quant_15/relux/Regularizer/mul/x:output:02activation_quant_15/relux/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/mul�
+activation_quant_15/relux/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+activation_quant_15/relux/Regularizer/add/x�
)activation_quant_15/relux/Regularizer/addAddV24activation_quant_15/relux/Regularizer/add/x:output:0-activation_quant_15/relux/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)activation_quant_15/relux/Regularizer/add�
IdentityIdentity!FakeQuantWithMinMaxVars:outputs:0'^FakeQuantWithMinMaxVars/ReadVariableOp^Minimum/ReadVariableOp<^activation_quant_15/relux/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:2P
&FakeQuantWithMinMaxVars/ReadVariableOp&FakeQuantWithMinMaxVars/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2z
;activation_quant_15/relux/Regularizer/Square/ReadVariableOp;activation_quant_15/relux/Regularizer/Square/ReadVariableOp:! 

_user_specified_namex
�$
�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_119932

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_119917
assignmovingavg_1_119924
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/119917*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/119917*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_119917*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/119917*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/119917*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_119917AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/119917*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/119924*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/119924*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_119924*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/119924*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/119924*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_119924AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/119924*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_118500

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�	
�
G__inference_dense_noise_layer_call_and_return_conditional_losses_120579
x"
matmul_readvariableop_resource
add_readvariableop_resource
identity��MatMul/ReadVariableOp�add/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOpn
MatMulMatMulxMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:
*
dtype02
add/ReadVariableOps
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
addX
SoftmaxSoftmaxadd:z:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:! 

_user_specified_namex
�
�
&__inference_model_layer_call_fn_119724
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25*%
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*/
config_proto

CPU

GPU2*0,1J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_1191082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������@:���������@::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_120048
x
abs_readvariableop_resource
readvariableop_1_resource
identity��Abs/ReadVariableOp�ReadVariableOp�ReadVariableOp_1�
Abs/ReadVariableOpReadVariableOpabs_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Abs/ReadVariableOp^
AbsAbsAbs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2
Absg
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
ConstK
MaxMaxAbs:y:0Const:output:0*
T0*
_output_shapes
: 2
MaxS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
mul/yP
mulMulMax:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mul�
random_normal/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      @   @   2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/mean�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*&
_output_shapes
:@@*
dtype02$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0mul:z:0*
T0*&
_output_shapes
:@@2
random_normal/mul�
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*&
_output_shapes
:@@2
random_normal�
ReadVariableOpReadVariableOpabs_readvariableop_resource^Abs/ReadVariableOp*&
_output_shapes
:@@*
dtype02
ReadVariableOpo
addAddV2ReadVariableOp:value:0random_normal:z:0*
T0*&
_output_shapes
:@@2
addW
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2	
mul_1/yV
mul_1MulMax:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1x
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:@2
random_normal_1/shapeq
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal_1/mean�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape:output:0*
T0*
_output_shapes
:@*
dtype02&
$random_normal_1/RandomStandardNormal�
random_normal_1/mulMul-random_normal_1/RandomStandardNormal:output:0	mul_1:z:0*
T0*
_output_shapes
:@2
random_normal_1/mul�
random_normal_1Addrandom_normal_1/mul:z:0random_normal_1/mean:output:0*
T0*
_output_shapes
:@2
random_normal_1z
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1k
add_1AddV2ReadVariableOp_1:value:0random_normal_1:z:0*
T0*
_output_shapes
:@2
add_1�
convolutionConv2Dxadd:z:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
convolutionx
BiasAddBiasAddconvolution:output:0	add_1:z:0*
T0*/
_output_shapes
:���������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^Abs/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::2(
Abs/ReadVariableOpAbs/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:! 

_user_specified_namex
�
�
7__inference_batch_normalization_17_layer_call_fn_120232

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

CPU

GPU2*0,1J 8*[
fVRT
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_1180172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_120118

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_120103
assignmovingavg_1_120110
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/120103*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/120103*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_120103*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/120103*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/120103*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_120103AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/120103*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/120110*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/120110*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_120110*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/120110*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/120110*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_120110AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/120110*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
0__inference_conv2d_noise_18_layer_call_fn_119805
x"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

CPU

GPU2*0,1J 8*T
fORM
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_1182332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:! 

_user_specified_namex
�
�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_120140

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_118118

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_118103
assignmovingavg_1_118110
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*)
_class
loc:@AssignMovingAvg/118103*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*)
_class
loc:@AssignMovingAvg/118103*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_118103*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*)
_class
loc:@AssignMovingAvg/118103*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*)
_class
loc:@AssignMovingAvg/118103*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_118103AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/118103*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst*+
_class!
loc:@AssignMovingAvg_1/118110*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/118110*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_118110*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*+
_class!
loc:@AssignMovingAvg_1/118110*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg_1/118110*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_118110AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/118110*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_120462

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������@
C
input_28
serving_default_input_2:0���������@?
dense_noise0
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer-12
layer-13
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
�_default_save_signature
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "input_spec": [null, null], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model"}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-06, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 8, 8, 64], "config": {"batch_input_shape": [null, 8, 8, 64], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
�
	relux
trainable_variables
regularization_losses
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "activation_quant", "name": "activation_quant_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
�

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "conv2d_noise", "name": "conv2d_noise_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 8, 8, 64], "config": {"batch_input_shape": [null, 8, 8, 64], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
�
#axis
	$gamma
%beta
&moving_mean
'moving_variance
(trainable_variables
)regularization_losses
*	variables
+	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_16", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
,trainable_variables
-regularization_losses
.	variables
/	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add", "trainable": true, "dtype": "float32"}}
�
	0relux
1trainable_variables
2regularization_losses
3	variables
4	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "activation_quant", "name": "activation_quant_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
�

5kernel
6bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "conv2d_noise", "name": "conv2d_noise_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
�
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_17", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
	Drelux
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "activation_quant", "name": "activation_quant_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
�

Ikernel
Jbias
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "conv2d_noise", "name": "conv2d_noise_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
�
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_18", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}}
�
\trainable_variables
]regularization_losses
^	variables
_	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "AveragePooling2D", "name": "average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [8, 8], "padding": "valid", "strides": [8, 8], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
`trainable_variables
aregularization_losses
b	variables
c	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�
	drelux
etrainable_variables
fregularization_losses
g	variables
h	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "activation_quant", "name": "activation_quant_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
�

ikernel
jbias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "dense_noise", "name": "dense_noise", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
�
oiter

pbeta_1

qbeta_2
	rdecay
slearning_ratem�m�m�$m�%m�0m�5m�6m�<m�=m�Dm�Im�Jm�Pm�Qm�dm�im�jm�v�v�v�$v�%v�0v�5v�6v�<v�=v�Dv�Iv�Jv�Pv�Qv�dv�iv�jv�"
	optimizer
�
0
1
2
$3
%4
05
56
67
<8
=9
D10
I11
J12
P13
Q14
d15
i16
j17"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
0
1
2
$3
%4
&5
'6
07
58
69
<10
=11
>12
?13
D14
I15
J16
P17
Q18
R19
S20
d21
i22
j23"
trackable_list_wrapper
�
tnon_trainable_variables
trainable_variables
umetrics
regularization_losses
vlayer_regularization_losses

wlayers
	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
#:! 2activation_quant_15/relux
'
0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
xnon_trainable_variables
trainable_variables
ymetrics
regularization_losses
zlayer_regularization_losses

{layers
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0:.@@2conv2d_noise_18/kernel
": @2conv2d_noise_18/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
|non_trainable_variables
trainable_variables
}metrics
 regularization_losses
~layer_regularization_losses

layers
!	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_16/gamma
):'@2batch_normalization_16/beta
2:0@ (2"batch_normalization_16/moving_mean
6:4@ (2&batch_normalization_16/moving_variance
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
$0
%1
&2
'3"
trackable_list_wrapper
�
�non_trainable_variables
(trainable_variables
�metrics
)regularization_losses
 �layer_regularization_losses
�layers
*	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
,trainable_variables
�metrics
-regularization_losses
 �layer_regularization_losses
�layers
.	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:! 2activation_quant_16/relux
'
00"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
'
00"
trackable_list_wrapper
�
�non_trainable_variables
1trainable_variables
�metrics
2regularization_losses
 �layer_regularization_losses
�layers
3	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0:.@@2conv2d_noise_19/kernel
": @2conv2d_noise_19/bias
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
�
�non_trainable_variables
7trainable_variables
�metrics
8regularization_losses
 �layer_regularization_losses
�layers
9	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_17/gamma
):'@2batch_normalization_17/beta
2:0@ (2"batch_normalization_17/moving_mean
6:4@ (2&batch_normalization_17/moving_variance
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
�
�non_trainable_variables
@trainable_variables
�metrics
Aregularization_losses
 �layer_regularization_losses
�layers
B	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:! 2activation_quant_17/relux
'
D0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
'
D0"
trackable_list_wrapper
�
�non_trainable_variables
Etrainable_variables
�metrics
Fregularization_losses
 �layer_regularization_losses
�layers
G	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0:.@@2conv2d_noise_20/kernel
": @2conv2d_noise_20/bias
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
�
�non_trainable_variables
Ktrainable_variables
�metrics
Lregularization_losses
 �layer_regularization_losses
�layers
M	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_18/gamma
):'@2batch_normalization_18/beta
2:0@ (2"batch_normalization_18/moving_mean
6:4@ (2&batch_normalization_18/moving_variance
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
P0
Q1
R2
S3"
trackable_list_wrapper
�
�non_trainable_variables
Ttrainable_variables
�metrics
Uregularization_losses
 �layer_regularization_losses
�layers
V	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
Xtrainable_variables
�metrics
Yregularization_losses
 �layer_regularization_losses
�layers
Z	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
\trainable_variables
�metrics
]regularization_losses
 �layer_regularization_losses
�layers
^	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
`trainable_variables
�metrics
aregularization_losses
 �layer_regularization_losses
�layers
b	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:! 2activation_quant_18/relux
'
d0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
'
d0"
trackable_list_wrapper
�
�non_trainable_variables
etrainable_variables
�metrics
fregularization_losses
 �layer_regularization_losses
�layers
g	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
$:"@
2dense_noise/kernel
:
2dense_noise/bias
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
�
�non_trainable_variables
ktrainable_variables
�metrics
lregularization_losses
 �layer_regularization_losses
�layers
m	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
J
&0
'1
>2
?3
R4
S5"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�
_fn_kwargs
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�trainable_variables
�metrics
�regularization_losses
 �layer_regularization_losses
�layers
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(:& 2 Adam/activation_quant_15/relux/m
5:3@@2Adam/conv2d_noise_18/kernel/m
':%@2Adam/conv2d_noise_18/bias/m
/:-@2#Adam/batch_normalization_16/gamma/m
.:,@2"Adam/batch_normalization_16/beta/m
(:& 2 Adam/activation_quant_16/relux/m
5:3@@2Adam/conv2d_noise_19/kernel/m
':%@2Adam/conv2d_noise_19/bias/m
/:-@2#Adam/batch_normalization_17/gamma/m
.:,@2"Adam/batch_normalization_17/beta/m
(:& 2 Adam/activation_quant_17/relux/m
5:3@@2Adam/conv2d_noise_20/kernel/m
':%@2Adam/conv2d_noise_20/bias/m
/:-@2#Adam/batch_normalization_18/gamma/m
.:,@2"Adam/batch_normalization_18/beta/m
(:& 2 Adam/activation_quant_18/relux/m
):'@
2Adam/dense_noise/kernel/m
#:!
2Adam/dense_noise/bias/m
(:& 2 Adam/activation_quant_15/relux/v
5:3@@2Adam/conv2d_noise_18/kernel/v
':%@2Adam/conv2d_noise_18/bias/v
/:-@2#Adam/batch_normalization_16/gamma/v
.:,@2"Adam/batch_normalization_16/beta/v
(:& 2 Adam/activation_quant_16/relux/v
5:3@@2Adam/conv2d_noise_19/kernel/v
':%@2Adam/conv2d_noise_19/bias/v
/:-@2#Adam/batch_normalization_17/gamma/v
.:,@2"Adam/batch_normalization_17/beta/v
(:& 2 Adam/activation_quant_17/relux/v
5:3@@2Adam/conv2d_noise_20/kernel/v
':%@2Adam/conv2d_noise_20/bias/v
/:-@2#Adam/batch_normalization_18/gamma/v
.:,@2"Adam/batch_normalization_18/beta/v
(:& 2 Adam/activation_quant_18/relux/v
):'@
2Adam/dense_noise/kernel/v
#:!
2Adam/dense_noise/bias/v
�2�
!__inference__wrapped_model_117760�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *^�[
Y�V
)�&
input_1���������@
)�&
input_2���������@
�2�
&__inference_model_layer_call_fn_119724
&__inference_model_layer_call_fn_119135
&__inference_model_layer_call_fn_119029
&__inference_model_layer_call_fn_119694�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_model_layer_call_and_return_conditional_losses_119664
A__inference_model_layer_call_and_return_conditional_losses_118922
A__inference_model_layer_call_and_return_conditional_losses_119511
A__inference_model_layer_call_and_return_conditional_losses_118846�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_activation_quant_15_layer_call_fn_119758�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_119752�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_conv2d_noise_18_layer_call_fn_119805
0__inference_conv2d_noise_18_layer_call_fn_119812�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_119788
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_119798�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_16_layer_call_fn_119963
7__inference_batch_normalization_16_layer_call_fn_119898
7__inference_batch_normalization_16_layer_call_fn_119972
7__inference_batch_normalization_16_layer_call_fn_119889�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_119932
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_119954
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_119858
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_119880�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
$__inference_add_layer_call_fn_119984�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_add_layer_call_and_return_conditional_losses_119978�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_activation_quant_16_layer_call_fn_120018�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_120012�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_conv2d_noise_19_layer_call_fn_120065
0__inference_conv2d_noise_19_layer_call_fn_120072�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_120058
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_120048�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_17_layer_call_fn_120223
7__inference_batch_normalization_17_layer_call_fn_120158
7__inference_batch_normalization_17_layer_call_fn_120149
7__inference_batch_normalization_17_layer_call_fn_120232�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_120214
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_120192
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_120118
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_120140�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_activation_quant_17_layer_call_fn_120266�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_120260�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_conv2d_noise_20_layer_call_fn_120313
0__inference_conv2d_noise_20_layer_call_fn_120320�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_120296
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_120306�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_batch_normalization_18_layer_call_fn_120406
7__inference_batch_normalization_18_layer_call_fn_120480
7__inference_batch_normalization_18_layer_call_fn_120397
7__inference_batch_normalization_18_layer_call_fn_120471�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_120462
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_120440
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_120366
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_120388�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_add_1_layer_call_fn_120492�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_add_1_layer_call_and_return_conditional_losses_120486�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
2__inference_average_pooling2d_layer_call_fn_118168�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_118162�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
(__inference_flatten_layer_call_fn_120503�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_flatten_layer_call_and_return_conditional_losses_120498�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_activation_quant_18_layer_call_fn_120537�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_120531�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_dense_noise_layer_call_fn_120586
,__inference_dense_noise_layer_call_fn_120593�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dense_noise_layer_call_and_return_conditional_losses_120568
G__inference_dense_noise_layer_call_and_return_conditional_losses_120579�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference_loss_fn_0_120606�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_120619�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_120632�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_120645�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
:B8
$__inference_signature_wrapper_119242input_1input_2
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
!__inference__wrapped_model_117760�$%&'056<=>?DIJPQRSdijh�e
^�[
Y�V
)�&
input_1���������@
)�&
input_2���������@
� "9�6
4
dense_noise%�"
dense_noise���������
�
O__inference_activation_quant_15_layer_call_and_return_conditional_losses_119752f2�/
(�%
#� 
x���������@
� "-�*
#� 
0���������@
� �
4__inference_activation_quant_15_layer_call_fn_119758Y2�/
(�%
#� 
x���������@
� " ����������@�
O__inference_activation_quant_16_layer_call_and_return_conditional_losses_120012f02�/
(�%
#� 
x���������@
� "-�*
#� 
0���������@
� �
4__inference_activation_quant_16_layer_call_fn_120018Y02�/
(�%
#� 
x���������@
� " ����������@�
O__inference_activation_quant_17_layer_call_and_return_conditional_losses_120260fD2�/
(�%
#� 
x���������@
� "-�*
#� 
0���������@
� �
4__inference_activation_quant_17_layer_call_fn_120266YD2�/
(�%
#� 
x���������@
� " ����������@�
O__inference_activation_quant_18_layer_call_and_return_conditional_losses_120531Vd*�'
 �
�
x���������@
� "%�"
�
0���������@
� �
4__inference_activation_quant_18_layer_call_fn_120537Id*�'
 �
�
x���������@
� "����������@�
A__inference_add_1_layer_call_and_return_conditional_losses_120486�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� "-�*
#� 
0���������@
� �
&__inference_add_1_layer_call_fn_120492�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� " ����������@�
?__inference_add_layer_call_and_return_conditional_losses_119978�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� "-�*
#� 
0���������@
� �
$__inference_add_layer_call_fn_119984�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� " ����������@�
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_118162�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
2__inference_average_pooling2d_layer_call_fn_118168�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_119858r$%&';�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_119880r$%&';�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_119932�$%&'M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_16_layer_call_and_return_conditional_losses_119954�$%&'M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
7__inference_batch_normalization_16_layer_call_fn_119889e$%&';�8
1�.
(�%
inputs���������@
p
� " ����������@�
7__inference_batch_normalization_16_layer_call_fn_119898e$%&';�8
1�.
(�%
inputs���������@
p 
� " ����������@�
7__inference_batch_normalization_16_layer_call_fn_119963�$%&'M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_16_layer_call_fn_119972�$%&'M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_120118r<=>?;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_120140r<=>?;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_120192�<=>?M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_17_layer_call_and_return_conditional_losses_120214�<=>?M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
7__inference_batch_normalization_17_layer_call_fn_120149e<=>?;�8
1�.
(�%
inputs���������@
p
� " ����������@�
7__inference_batch_normalization_17_layer_call_fn_120158e<=>?;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
7__inference_batch_normalization_17_layer_call_fn_120223�<=>?M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_17_layer_call_fn_120232�<=>?M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_120366�PQRSM�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_120388�PQRSM�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_120440rPQRS;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
R__inference_batch_normalization_18_layer_call_and_return_conditional_losses_120462rPQRS;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
7__inference_batch_normalization_18_layer_call_fn_120397�PQRSM�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
7__inference_batch_normalization_18_layer_call_fn_120406�PQRSM�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
7__inference_batch_normalization_18_layer_call_fn_120471ePQRS;�8
1�.
(�%
inputs���������@
p
� " ����������@�
7__inference_batch_normalization_18_layer_call_fn_120480ePQRS;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_119788k6�3
,�)
#� 
x���������@
p
� "-�*
#� 
0���������@
� �
K__inference_conv2d_noise_18_layer_call_and_return_conditional_losses_119798k6�3
,�)
#� 
x���������@
p 
� "-�*
#� 
0���������@
� �
0__inference_conv2d_noise_18_layer_call_fn_119805^6�3
,�)
#� 
x���������@
p
� " ����������@�
0__inference_conv2d_noise_18_layer_call_fn_119812^6�3
,�)
#� 
x���������@
p 
� " ����������@�
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_120048k566�3
,�)
#� 
x���������@
p
� "-�*
#� 
0���������@
� �
K__inference_conv2d_noise_19_layer_call_and_return_conditional_losses_120058k566�3
,�)
#� 
x���������@
p 
� "-�*
#� 
0���������@
� �
0__inference_conv2d_noise_19_layer_call_fn_120065^566�3
,�)
#� 
x���������@
p
� " ����������@�
0__inference_conv2d_noise_19_layer_call_fn_120072^566�3
,�)
#� 
x���������@
p 
� " ����������@�
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_120296kIJ6�3
,�)
#� 
x���������@
p
� "-�*
#� 
0���������@
� �
K__inference_conv2d_noise_20_layer_call_and_return_conditional_losses_120306kIJ6�3
,�)
#� 
x���������@
p 
� "-�*
#� 
0���������@
� �
0__inference_conv2d_noise_20_layer_call_fn_120313^IJ6�3
,�)
#� 
x���������@
p
� " ����������@�
0__inference_conv2d_noise_20_layer_call_fn_120320^IJ6�3
,�)
#� 
x���������@
p 
� " ����������@�
G__inference_dense_noise_layer_call_and_return_conditional_losses_120568[ij.�+
$�!
�
x���������@
p
� "%�"
�
0���������

� �
G__inference_dense_noise_layer_call_and_return_conditional_losses_120579[ij.�+
$�!
�
x���������@
p 
� "%�"
�
0���������

� ~
,__inference_dense_noise_layer_call_fn_120586Nij.�+
$�!
�
x���������@
p
� "����������
~
,__inference_dense_noise_layer_call_fn_120593Nij.�+
$�!
�
x���������@
p 
� "����������
�
C__inference_flatten_layer_call_and_return_conditional_losses_120498`7�4
-�*
(�%
inputs���������@
� "%�"
�
0���������@
� 
(__inference_flatten_layer_call_fn_120503S7�4
-�*
(�%
inputs���������@
� "����������@;
__inference_loss_fn_0_120606�

� 
� "� ;
__inference_loss_fn_1_1206190�

� 
� "� ;
__inference_loss_fn_2_120632D�

� 
� "� ;
__inference_loss_fn_3_120645d�

� 
� "� �
A__inference_model_layer_call_and_return_conditional_losses_118846�$%&'056<=>?DIJPQRSdijp�m
f�c
Y�V
)�&
input_1���������@
)�&
input_2���������@
p

 
� "%�"
�
0���������

� �
A__inference_model_layer_call_and_return_conditional_losses_118922�$%&'056<=>?DIJPQRSdijp�m
f�c
Y�V
)�&
input_1���������@
)�&
input_2���������@
p 

 
� "%�"
�
0���������

� �
A__inference_model_layer_call_and_return_conditional_losses_119511�$%&'056<=>?DIJPQRSdijr�o
h�e
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
p

 
� "%�"
�
0���������

� �
A__inference_model_layer_call_and_return_conditional_losses_119664�$%&'056<=>?DIJPQRSdijr�o
h�e
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
p 

 
� "%�"
�
0���������

� �
&__inference_model_layer_call_fn_119029�$%&'056<=>?DIJPQRSdijp�m
f�c
Y�V
)�&
input_1���������@
)�&
input_2���������@
p

 
� "����������
�
&__inference_model_layer_call_fn_119135�$%&'056<=>?DIJPQRSdijp�m
f�c
Y�V
)�&
input_1���������@
)�&
input_2���������@
p 

 
� "����������
�
&__inference_model_layer_call_fn_119694�$%&'056<=>?DIJPQRSdijr�o
h�e
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
p

 
� "����������
�
&__inference_model_layer_call_fn_119724�$%&'056<=>?DIJPQRSdijr�o
h�e
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
p 

 
� "����������
�
$__inference_signature_wrapper_119242�$%&'056<=>?DIJPQRSdijy�v
� 
o�l
4
input_1)�&
input_1���������@
4
input_2)�&
input_2���������@"9�6
4
dense_noise%�"
dense_noise���������
