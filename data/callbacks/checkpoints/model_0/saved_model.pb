ò
Ý
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68û
|
dense_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô* 
shared_namedense_56/kernel
u
#dense_56/kernel/Read/ReadVariableOpReadVariableOpdense_56/kernel* 
_output_shapes
:
ô*
dtype0
s
dense_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô*
shared_namedense_56/bias
l
!dense_56/bias/Read/ReadVariableOpReadVariableOpdense_56/bias*
_output_shapes	
:ô*
dtype0
|
dense_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôô* 
shared_namedense_57/kernel
u
#dense_57/kernel/Read/ReadVariableOpReadVariableOpdense_57/kernel* 
_output_shapes
:
ôô*
dtype0
s
dense_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô*
shared_namedense_57/bias
l
!dense_57/bias/Read/ReadVariableOpReadVariableOpdense_57/bias*
_output_shapes	
:ô*
dtype0
|
dense_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôô* 
shared_namedense_58/kernel
u
#dense_58/kernel/Read/ReadVariableOpReadVariableOpdense_58/kernel* 
_output_shapes
:
ôô*
dtype0
s
dense_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô*
shared_namedense_58/bias
l
!dense_58/bias/Read/ReadVariableOpReadVariableOpdense_58/bias*
_output_shapes	
:ô*
dtype0
{
dense_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ô
* 
shared_namedense_59/kernel
t
#dense_59/kernel/Read/ReadVariableOpReadVariableOpdense_59/kernel*
_output_shapes
:	ô
*
dtype0
r
dense_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_59/bias
k
!dense_59/bias/Read/ReadVariableOpReadVariableOpdense_59/bias*
_output_shapes
:
*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/dense_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô*'
shared_nameAdam/dense_56/kernel/m

*Adam/dense_56/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/m* 
_output_shapes
:
ô*
dtype0

Adam/dense_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô*%
shared_nameAdam/dense_56/bias/m
z
(Adam/dense_56/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/m*
_output_shapes	
:ô*
dtype0

Adam/dense_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôô*'
shared_nameAdam/dense_57/kernel/m

*Adam/dense_57/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/m* 
_output_shapes
:
ôô*
dtype0

Adam/dense_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô*%
shared_nameAdam/dense_57/bias/m
z
(Adam/dense_57/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/m*
_output_shapes	
:ô*
dtype0

Adam/dense_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôô*'
shared_nameAdam/dense_58/kernel/m

*Adam/dense_58/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/m* 
_output_shapes
:
ôô*
dtype0

Adam/dense_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô*%
shared_nameAdam/dense_58/bias/m
z
(Adam/dense_58/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/m*
_output_shapes	
:ô*
dtype0

Adam/dense_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ô
*'
shared_nameAdam/dense_59/kernel/m

*Adam/dense_59/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/m*
_output_shapes
:	ô
*
dtype0

Adam/dense_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_59/bias/m
y
(Adam/dense_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/m*
_output_shapes
:
*
dtype0

Adam/dense_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ô*'
shared_nameAdam/dense_56/kernel/v

*Adam/dense_56/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/kernel/v* 
_output_shapes
:
ô*
dtype0

Adam/dense_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô*%
shared_nameAdam/dense_56/bias/v
z
(Adam/dense_56/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_56/bias/v*
_output_shapes	
:ô*
dtype0

Adam/dense_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôô*'
shared_nameAdam/dense_57/kernel/v

*Adam/dense_57/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/kernel/v* 
_output_shapes
:
ôô*
dtype0

Adam/dense_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô*%
shared_nameAdam/dense_57/bias/v
z
(Adam/dense_57/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_57/bias/v*
_output_shapes	
:ô*
dtype0

Adam/dense_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôô*'
shared_nameAdam/dense_58/kernel/v

*Adam/dense_58/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/kernel/v* 
_output_shapes
:
ôô*
dtype0

Adam/dense_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ô*%
shared_nameAdam/dense_58/bias/v
z
(Adam/dense_58/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_58/bias/v*
_output_shapes	
:ô*
dtype0

Adam/dense_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ô
*'
shared_nameAdam/dense_59/kernel/v

*Adam/dense_59/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/kernel/v*
_output_shapes
:	ô
*
dtype0

Adam/dense_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_59/bias/v
y
(Adam/dense_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_59/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
±8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ì7
valueâ7Bß7 BØ7
õ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
¦

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
Ô

/beta_1

0beta_2
	1decay
2learning_rate
3iterm]m^m_m`ma mb'mc(mdvevfvgvhvi vj'vk(vl*
<
0
1
2
3
4
 5
'6
(7*
<
0
1
2
3
4
 5
'6
(7*

40
51
62
73* 
°
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

=serving_default* 
_Y
VARIABLE_VALUEdense_56/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_56/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
	
40* 

>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_57/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_57/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
	
50* 

Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_58/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_58/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
	
60* 

Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_59/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_59/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
	
70* 

Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
'
0
1
2
3
4*

R0
S1*
* 
* 
* 
* 
* 
* 
	
40* 
* 
* 
* 
* 
	
50* 
* 
* 
* 
* 
	
60* 
* 
* 
* 
* 
	
70* 
* 
8
	Ttotal
	Ucount
V	variables
W	keras_api*
H
	Xtotal
	Ycount
Z
_fn_kwargs
[	variables
\	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

T0
U1*

V	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

X0
Y1*

[	variables*
|
VARIABLE_VALUEAdam/dense_56/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_56/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_57/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_57/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_58/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_58/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_59/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_59/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_56/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_56/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_57/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_57/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_58/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_58/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_59/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_59/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
serving_default_input_15Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ã
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_15dense_56/kerneldense_56/biasdense_57/kerneldense_57/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_268756
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_56/kernel/Read/ReadVariableOp!dense_56/bias/Read/ReadVariableOp#dense_57/kernel/Read/ReadVariableOp!dense_57/bias/Read/ReadVariableOp#dense_58/kernel/Read/ReadVariableOp!dense_58/bias/Read/ReadVariableOp#dense_59/kernel/Read/ReadVariableOp!dense_59/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_56/kernel/m/Read/ReadVariableOp(Adam/dense_56/bias/m/Read/ReadVariableOp*Adam/dense_57/kernel/m/Read/ReadVariableOp(Adam/dense_57/bias/m/Read/ReadVariableOp*Adam/dense_58/kernel/m/Read/ReadVariableOp(Adam/dense_58/bias/m/Read/ReadVariableOp*Adam/dense_59/kernel/m/Read/ReadVariableOp(Adam/dense_59/bias/m/Read/ReadVariableOp*Adam/dense_56/kernel/v/Read/ReadVariableOp(Adam/dense_56/bias/v/Read/ReadVariableOp*Adam/dense_57/kernel/v/Read/ReadVariableOp(Adam/dense_57/bias/v/Read/ReadVariableOp*Adam/dense_58/kernel/v/Read/ReadVariableOp(Adam/dense_58/bias/v/Read/ReadVariableOp*Adam/dense_59/kernel/v/Read/ReadVariableOp(Adam/dense_59/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_269050
ø
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_56/kerneldense_56/biasdense_57/kerneldense_57/biasdense_58/kerneldense_58/biasdense_59/kerneldense_59/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcounttotal_1count_1Adam/dense_56/kernel/mAdam/dense_56/bias/mAdam/dense_57/kernel/mAdam/dense_57/bias/mAdam/dense_58/kernel/mAdam/dense_58/bias/mAdam/dense_59/kernel/mAdam/dense_59/bias/mAdam/dense_56/kernel/vAdam/dense_56/bias/vAdam/dense_57/kernel/vAdam/dense_57/bias/vAdam/dense_58/kernel/vAdam/dense_58/bias/vAdam/dense_59/kernel/vAdam/dense_59/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_269159ìì
¼
¬
D__inference_dense_58_layer_call_and_return_conditional_losses_268229

inputs2
matmul_readvariableop_resource:
ôô.
biasadd_readvariableop_resource:	ô
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_58/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
1dense_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
"dense_58/kernel/Regularizer/SquareSquare9dense_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum&dense_58/kernel/Regularizer/Square:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_58/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_58/kernel/Regularizer/Square/ReadVariableOp1dense_58/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
º
²
__inference_loss_fn_3_268928M
:dense_59_kernel_regularizer_square_readvariableop_resource:	ô

identity¢1dense_59/kernel/Regularizer/Square/ReadVariableOp­
1dense_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_59_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	ô
*
dtype0
"dense_59/kernel/Regularizer/SquareSquare9dense_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô
r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum&dense_59/kernel/Regularizer/Square:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_59/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_59/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_59/kernel/Regularizer/Square/ReadVariableOp1dense_59/kernel/Regularizer/Square/ReadVariableOp
Ê	
Â
)__inference_model_14_layer_call_fn_268621

inputs
unknown:
ô
	unknown_0:	ô
	unknown_1:
ôô
	unknown_2:	ô
	unknown_3:
ôô
	unknown_4:	ô
	unknown_5:	ô

	unknown_6:

identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_268413o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
¬
D__inference_dense_56_layer_call_and_return_conditional_losses_268788

inputs2
matmul_readvariableop_resource:
ô.
biasadd_readvariableop_resource:	ô
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_56/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
1dense_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0
"dense_56/kernel/Regularizer/SquareSquare9dense_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôr
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum&dense_56/kernel/Regularizer/Square:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_56/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_56/kernel/Regularizer/Square/ReadVariableOp1dense_56/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
¬
D__inference_dense_57_layer_call_and_return_conditional_losses_268820

inputs2
matmul_readvariableop_resource:
ôô.
biasadd_readvariableop_resource:	ô
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_57/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
1dense_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
"dense_57/kernel/Regularizer/SquareSquare9dense_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum&dense_57/kernel/Regularizer/Square:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_57/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_57/kernel/Regularizer/Square/ReadVariableOp1dense_57/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
É

)__inference_dense_57_layer_call_fn_268803

inputs
unknown:
ôô
	unknown_0:	ô
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_268206p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
·
ª
D__inference_dense_59_layer_call_and_return_conditional_losses_268252

inputs1
matmul_readvariableop_resource:	ô
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_59/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ô
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

1dense_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ô
*
dtype0
"dense_59/kernel/Regularizer/SquareSquare9dense_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô
r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum&dense_59/kernel/Regularizer/Square:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_59/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_59/kernel/Regularizer/Square/ReadVariableOp1dense_59/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
Ð	
Ä
)__inference_model_14_layer_call_fn_268453
input_15
unknown:
ô
	unknown_0:	ô
	unknown_1:
ôô
	unknown_2:	ô
	unknown_3:
ôô
	unknown_4:	ô
	unknown_5:	ô

	unknown_6:

identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinput_15unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_268413o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_15
¨	
¿
$__inference_signature_wrapper_268756
input_15
unknown:
ô
	unknown_0:	ô
	unknown_1:
ôô
	unknown_2:	ô
	unknown_3:
ôô
	unknown_4:	ô
	unknown_5:	ô

	unknown_6:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_15unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_268159o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_15
F
¬
__inference__traced_save_269050
file_prefix.
*savev2_dense_56_kernel_read_readvariableop,
(savev2_dense_56_bias_read_readvariableop.
*savev2_dense_57_kernel_read_readvariableop,
(savev2_dense_57_bias_read_readvariableop.
*savev2_dense_58_kernel_read_readvariableop,
(savev2_dense_58_bias_read_readvariableop.
*savev2_dense_59_kernel_read_readvariableop,
(savev2_dense_59_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_56_kernel_m_read_readvariableop3
/savev2_adam_dense_56_bias_m_read_readvariableop5
1savev2_adam_dense_57_kernel_m_read_readvariableop3
/savev2_adam_dense_57_bias_m_read_readvariableop5
1savev2_adam_dense_58_kernel_m_read_readvariableop3
/savev2_adam_dense_58_bias_m_read_readvariableop5
1savev2_adam_dense_59_kernel_m_read_readvariableop3
/savev2_adam_dense_59_bias_m_read_readvariableop5
1savev2_adam_dense_56_kernel_v_read_readvariableop3
/savev2_adam_dense_56_bias_v_read_readvariableop5
1savev2_adam_dense_57_kernel_v_read_readvariableop3
/savev2_adam_dense_57_bias_v_read_readvariableop5
1savev2_adam_dense_58_kernel_v_read_readvariableop3
/savev2_adam_dense_58_bias_v_read_readvariableop5
1savev2_adam_dense_59_kernel_v_read_readvariableop3
/savev2_adam_dense_59_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¯
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ø
valueÎBË"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH±
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_56_kernel_read_readvariableop(savev2_dense_56_bias_read_readvariableop*savev2_dense_57_kernel_read_readvariableop(savev2_dense_57_bias_read_readvariableop*savev2_dense_58_kernel_read_readvariableop(savev2_dense_58_bias_read_readvariableop*savev2_dense_59_kernel_read_readvariableop(savev2_dense_59_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_56_kernel_m_read_readvariableop/savev2_adam_dense_56_bias_m_read_readvariableop1savev2_adam_dense_57_kernel_m_read_readvariableop/savev2_adam_dense_57_bias_m_read_readvariableop1savev2_adam_dense_58_kernel_m_read_readvariableop/savev2_adam_dense_58_bias_m_read_readvariableop1savev2_adam_dense_59_kernel_m_read_readvariableop/savev2_adam_dense_59_bias_m_read_readvariableop1savev2_adam_dense_56_kernel_v_read_readvariableop/savev2_adam_dense_56_bias_v_read_readvariableop1savev2_adam_dense_57_kernel_v_read_readvariableop/savev2_adam_dense_57_bias_v_read_readvariableop1savev2_adam_dense_58_kernel_v_read_readvariableop/savev2_adam_dense_58_bias_v_read_readvariableop1savev2_adam_dense_59_kernel_v_read_readvariableop/savev2_adam_dense_59_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes÷
ô: :
ô:ô:
ôô:ô:
ôô:ô:	ô
:
: : : : : : : : : :
ô:ô:
ôô:ô:
ôô:ô:	ô
:
:
ô:ô:
ôô:ô:
ôô:ô:	ô
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
ô:!

_output_shapes	
:ô:&"
 
_output_shapes
:
ôô:!

_output_shapes	
:ô:&"
 
_output_shapes
:
ôô:!

_output_shapes	
:ô:%!

_output_shapes
:	ô
: 

_output_shapes
:
:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
ô:!

_output_shapes	
:ô:&"
 
_output_shapes
:
ôô:!

_output_shapes	
:ô:&"
 
_output_shapes
:
ôô:!

_output_shapes	
:ô:%!

_output_shapes
:	ô
: 

_output_shapes
:
:&"
 
_output_shapes
:
ô:!

_output_shapes	
:ô:&"
 
_output_shapes
:
ôô:!

_output_shapes	
:ô:&"
 
_output_shapes
:
ôô:!

_output_shapes	
:ô:% !

_output_shapes
:	ô
: !

_output_shapes
:
:"

_output_shapes
: 
ÕE

D__inference_model_14_layer_call_and_return_conditional_losses_268733

inputs;
'dense_56_matmul_readvariableop_resource:
ô7
(dense_56_biasadd_readvariableop_resource:	ô;
'dense_57_matmul_readvariableop_resource:
ôô7
(dense_57_biasadd_readvariableop_resource:	ô;
'dense_58_matmul_readvariableop_resource:
ôô7
(dense_58_biasadd_readvariableop_resource:	ô:
'dense_59_matmul_readvariableop_resource:	ô
6
(dense_59_biasadd_readvariableop_resource:

identity¢dense_56/BiasAdd/ReadVariableOp¢dense_56/MatMul/ReadVariableOp¢1dense_56/kernel/Regularizer/Square/ReadVariableOp¢dense_57/BiasAdd/ReadVariableOp¢dense_57/MatMul/ReadVariableOp¢1dense_57/kernel/Regularizer/Square/ReadVariableOp¢dense_58/BiasAdd/ReadVariableOp¢dense_58/MatMul/ReadVariableOp¢1dense_58/kernel/Regularizer/Square/ReadVariableOp¢dense_59/BiasAdd/ReadVariableOp¢dense_59/MatMul/ReadVariableOp¢1dense_59/kernel/Regularizer/Square/ReadVariableOp
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0|
dense_56/MatMulMatMulinputs&dense_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôc
dense_56/ReluReludense_56/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
dense_57/MatMulMatMuldense_56/Relu:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôc
dense_57/ReluReludense_57/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
dense_58/MatMulMatMuldense_57/Relu:activations:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôc
dense_58/ReluReludense_58/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes
:	ô
*
dtype0
dense_59/MatMulMatMuldense_58/Relu:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
dense_59/SoftmaxSoftmaxdense_59/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

1dense_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0
"dense_56/kernel/Regularizer/SquareSquare9dense_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôr
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum&dense_56/kernel/Regularizer/Square:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
"dense_57/kernel/Regularizer/SquareSquare9dense_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum&dense_57/kernel/Regularizer/Square:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
"dense_58/kernel/Regularizer/SquareSquare9dense_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum&dense_58/kernel/Regularizer/Square:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes
:	ô
*
dtype0
"dense_59/kernel/Regularizer/SquareSquare9dense_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô
r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum&dense_59/kernel/Regularizer/Square:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_59/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
NoOpNoOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp2^dense_56/kernel/Regularizer/Square/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp2^dense_57/kernel/Regularizer/Square/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp2^dense_58/kernel/Regularizer/Square/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp2^dense_59/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2f
1dense_56/kernel/Regularizer/Square/ReadVariableOp1dense_56/kernel/Regularizer/Square/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2f
1dense_57/kernel/Regularizer/Square/ReadVariableOp1dense_57/kernel/Regularizer/Square/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2f
1dense_58/kernel/Regularizer/Square/ReadVariableOp1dense_58/kernel/Regularizer/Square/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp2f
1dense_59/kernel/Regularizer/Square/ReadVariableOp1dense_59/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
Ô
"__inference__traced_restore_269159
file_prefix4
 assignvariableop_dense_56_kernel:
ô/
 assignvariableop_1_dense_56_bias:	ô6
"assignvariableop_2_dense_57_kernel:
ôô/
 assignvariableop_3_dense_57_bias:	ô6
"assignvariableop_4_dense_58_kernel:
ôô/
 assignvariableop_5_dense_58_bias:	ô5
"assignvariableop_6_dense_59_kernel:	ô
.
 assignvariableop_7_dense_59_bias:
#
assignvariableop_8_beta_1: #
assignvariableop_9_beta_2: #
assignvariableop_10_decay: +
!assignvariableop_11_learning_rate: '
assignvariableop_12_adam_iter:	 #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: >
*assignvariableop_17_adam_dense_56_kernel_m:
ô7
(assignvariableop_18_adam_dense_56_bias_m:	ô>
*assignvariableop_19_adam_dense_57_kernel_m:
ôô7
(assignvariableop_20_adam_dense_57_bias_m:	ô>
*assignvariableop_21_adam_dense_58_kernel_m:
ôô7
(assignvariableop_22_adam_dense_58_bias_m:	ô=
*assignvariableop_23_adam_dense_59_kernel_m:	ô
6
(assignvariableop_24_adam_dense_59_bias_m:
>
*assignvariableop_25_adam_dense_56_kernel_v:
ô7
(assignvariableop_26_adam_dense_56_bias_v:	ô>
*assignvariableop_27_adam_dense_57_kernel_v:
ôô7
(assignvariableop_28_adam_dense_57_bias_v:	ô>
*assignvariableop_29_adam_dense_58_kernel_v:
ôô7
(assignvariableop_30_adam_dense_58_bias_v:	ô=
*assignvariableop_31_adam_dense_59_kernel_v:	ô
6
(assignvariableop_32_adam_dense_59_bias_v:

identity_34¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9²
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ø
valueÎBË"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ë
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_56_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_56_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_57_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_57_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_58_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_58_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_59_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_59_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_beta_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_beta_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_decayIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp!assignvariableop_11_learning_rateIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_56_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_56_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_57_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_57_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_58_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_58_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_59_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_59_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_56_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_56_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_57_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_57_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_58_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_58_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_59_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_59_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¥
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¼
¬
D__inference_dense_57_layer_call_and_return_conditional_losses_268206

inputs2
matmul_readvariableop_resource:
ôô.
biasadd_readvariableop_resource:	ô
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_57/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
1dense_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
"dense_57/kernel/Regularizer/SquareSquare9dense_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum&dense_57/kernel/Regularizer/Square:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_57/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_57/kernel/Regularizer/Square/ReadVariableOp1dense_57/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
¼
¬
D__inference_dense_56_layer_call_and_return_conditional_losses_268183

inputs2
matmul_readvariableop_resource:
ô.
biasadd_readvariableop_resource:	ô
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_56/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
1dense_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0
"dense_56/kernel/Regularizer/SquareSquare9dense_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôr
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum&dense_56/kernel/Regularizer/Square:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_56/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_56/kernel/Regularizer/Square/ReadVariableOp1dense_56/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÕE

D__inference_model_14_layer_call_and_return_conditional_losses_268677

inputs;
'dense_56_matmul_readvariableop_resource:
ô7
(dense_56_biasadd_readvariableop_resource:	ô;
'dense_57_matmul_readvariableop_resource:
ôô7
(dense_57_biasadd_readvariableop_resource:	ô;
'dense_58_matmul_readvariableop_resource:
ôô7
(dense_58_biasadd_readvariableop_resource:	ô:
'dense_59_matmul_readvariableop_resource:	ô
6
(dense_59_biasadd_readvariableop_resource:

identity¢dense_56/BiasAdd/ReadVariableOp¢dense_56/MatMul/ReadVariableOp¢1dense_56/kernel/Regularizer/Square/ReadVariableOp¢dense_57/BiasAdd/ReadVariableOp¢dense_57/MatMul/ReadVariableOp¢1dense_57/kernel/Regularizer/Square/ReadVariableOp¢dense_58/BiasAdd/ReadVariableOp¢dense_58/MatMul/ReadVariableOp¢1dense_58/kernel/Regularizer/Square/ReadVariableOp¢dense_59/BiasAdd/ReadVariableOp¢dense_59/MatMul/ReadVariableOp¢1dense_59/kernel/Regularizer/Square/ReadVariableOp
dense_56/MatMul/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0|
dense_56/MatMulMatMulinputs&dense_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
dense_56/BiasAdd/ReadVariableOpReadVariableOp(dense_56_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôc
dense_56/ReluReludense_56/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
dense_57/MatMul/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
dense_57/MatMulMatMuldense_56/Relu:activations:0&dense_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
dense_57/BiasAdd/ReadVariableOpReadVariableOp(dense_57_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0
dense_57/BiasAddBiasAdddense_57/MatMul:product:0'dense_57/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôc
dense_57/ReluReludense_57/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
dense_58/MatMul/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
dense_58/MatMulMatMuldense_57/Relu:activations:0&dense_58/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
dense_58/BiasAdd/ReadVariableOpReadVariableOp(dense_58_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0
dense_58/BiasAddBiasAdddense_58/MatMul:product:0'dense_58/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôc
dense_58/ReluReludense_58/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
dense_59/MatMul/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes
:	ô
*
dtype0
dense_59/MatMulMatMuldense_58/Relu:activations:0&dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dense_59/BiasAdd/ReadVariableOpReadVariableOp(dense_59_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_59/BiasAddBiasAdddense_59/MatMul:product:0'dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
dense_59/SoftmaxSoftmaxdense_59/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

1dense_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_56_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0
"dense_56/kernel/Regularizer/SquareSquare9dense_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôr
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum&dense_56/kernel/Regularizer/Square:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_57_matmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
"dense_57/kernel/Regularizer/SquareSquare9dense_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum&dense_57/kernel/Regularizer/Square:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_58_matmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
"dense_58/kernel/Regularizer/SquareSquare9dense_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum&dense_58/kernel/Regularizer/Square:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_59_matmul_readvariableop_resource*
_output_shapes
:	ô
*
dtype0
"dense_59/kernel/Regularizer/SquareSquare9dense_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô
r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum&dense_59/kernel/Regularizer/Square:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_59/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
NoOpNoOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp2^dense_56/kernel/Regularizer/Square/ReadVariableOp ^dense_57/BiasAdd/ReadVariableOp^dense_57/MatMul/ReadVariableOp2^dense_57/kernel/Regularizer/Square/ReadVariableOp ^dense_58/BiasAdd/ReadVariableOp^dense_58/MatMul/ReadVariableOp2^dense_58/kernel/Regularizer/Square/ReadVariableOp ^dense_59/BiasAdd/ReadVariableOp^dense_59/MatMul/ReadVariableOp2^dense_59/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp2f
1dense_56/kernel/Regularizer/Square/ReadVariableOp1dense_56/kernel/Regularizer/Square/ReadVariableOp2B
dense_57/BiasAdd/ReadVariableOpdense_57/BiasAdd/ReadVariableOp2@
dense_57/MatMul/ReadVariableOpdense_57/MatMul/ReadVariableOp2f
1dense_57/kernel/Regularizer/Square/ReadVariableOp1dense_57/kernel/Regularizer/Square/ReadVariableOp2B
dense_58/BiasAdd/ReadVariableOpdense_58/BiasAdd/ReadVariableOp2@
dense_58/MatMul/ReadVariableOpdense_58/MatMul/ReadVariableOp2f
1dense_58/kernel/Regularizer/Square/ReadVariableOp1dense_58/kernel/Regularizer/Square/ReadVariableOp2B
dense_59/BiasAdd/ReadVariableOpdense_59/BiasAdd/ReadVariableOp2@
dense_59/MatMul/ReadVariableOpdense_59/MatMul/ReadVariableOp2f
1dense_59/kernel/Regularizer/Square/ReadVariableOp1dense_59/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

)__inference_dense_58_layer_call_fn_268835

inputs
unknown:
ôô
	unknown_0:	ô
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_268229p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
·
ª
D__inference_dense_59_layer_call_and_return_conditional_losses_268884

inputs1
matmul_readvariableop_resource:	ô
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_59/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ô
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

1dense_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ô
*
dtype0
"dense_59/kernel/Regularizer/SquareSquare9dense_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô
r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum&dense_59/kernel/Regularizer/Square:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_59/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_59/kernel/Regularizer/Square/ReadVariableOp1dense_59/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
Ê	
Â
)__inference_model_14_layer_call_fn_268600

inputs
unknown:
ô
	unknown_0:	ô
	unknown_1:
ôô
	unknown_2:	ô
	unknown_3:
ôô
	unknown_4:	ô
	unknown_5:	ô

	unknown_6:

identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_268283o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß*
´
!__inference__wrapped_model_268159
input_15D
0model_14_dense_56_matmul_readvariableop_resource:
ô@
1model_14_dense_56_biasadd_readvariableop_resource:	ôD
0model_14_dense_57_matmul_readvariableop_resource:
ôô@
1model_14_dense_57_biasadd_readvariableop_resource:	ôD
0model_14_dense_58_matmul_readvariableop_resource:
ôô@
1model_14_dense_58_biasadd_readvariableop_resource:	ôC
0model_14_dense_59_matmul_readvariableop_resource:	ô
?
1model_14_dense_59_biasadd_readvariableop_resource:

identity¢(model_14/dense_56/BiasAdd/ReadVariableOp¢'model_14/dense_56/MatMul/ReadVariableOp¢(model_14/dense_57/BiasAdd/ReadVariableOp¢'model_14/dense_57/MatMul/ReadVariableOp¢(model_14/dense_58/BiasAdd/ReadVariableOp¢'model_14/dense_58/MatMul/ReadVariableOp¢(model_14/dense_59/BiasAdd/ReadVariableOp¢'model_14/dense_59/MatMul/ReadVariableOp
'model_14/dense_56/MatMul/ReadVariableOpReadVariableOp0model_14_dense_56_matmul_readvariableop_resource* 
_output_shapes
:
ô*
dtype0
model_14/dense_56/MatMulMatMulinput_15/model_14/dense_56/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
(model_14/dense_56/BiasAdd/ReadVariableOpReadVariableOp1model_14_dense_56_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0­
model_14/dense_56/BiasAddBiasAdd"model_14/dense_56/MatMul:product:00model_14/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôu
model_14/dense_56/ReluRelu"model_14/dense_56/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
'model_14/dense_57/MatMul/ReadVariableOpReadVariableOp0model_14_dense_57_matmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0¬
model_14/dense_57/MatMulMatMul$model_14/dense_56/Relu:activations:0/model_14/dense_57/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
(model_14/dense_57/BiasAdd/ReadVariableOpReadVariableOp1model_14_dense_57_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0­
model_14/dense_57/BiasAddBiasAdd"model_14/dense_57/MatMul:product:00model_14/dense_57/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôu
model_14/dense_57/ReluRelu"model_14/dense_57/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
'model_14/dense_58/MatMul/ReadVariableOpReadVariableOp0model_14_dense_58_matmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0¬
model_14/dense_58/MatMulMatMul$model_14/dense_57/Relu:activations:0/model_14/dense_58/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
(model_14/dense_58/BiasAdd/ReadVariableOpReadVariableOp1model_14_dense_58_biasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0­
model_14/dense_58/BiasAddBiasAdd"model_14/dense_58/MatMul:product:00model_14/dense_58/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôu
model_14/dense_58/ReluRelu"model_14/dense_58/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
'model_14/dense_59/MatMul/ReadVariableOpReadVariableOp0model_14_dense_59_matmul_readvariableop_resource*
_output_shapes
:	ô
*
dtype0«
model_14/dense_59/MatMulMatMul$model_14/dense_58/Relu:activations:0/model_14/dense_59/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(model_14/dense_59/BiasAdd/ReadVariableOpReadVariableOp1model_14_dense_59_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0¬
model_14/dense_59/BiasAddBiasAdd"model_14/dense_59/MatMul:product:00model_14/dense_59/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
model_14/dense_59/SoftmaxSoftmax"model_14/dense_59/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
IdentityIdentity#model_14/dense_59/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp)^model_14/dense_56/BiasAdd/ReadVariableOp(^model_14/dense_56/MatMul/ReadVariableOp)^model_14/dense_57/BiasAdd/ReadVariableOp(^model_14/dense_57/MatMul/ReadVariableOp)^model_14/dense_58/BiasAdd/ReadVariableOp(^model_14/dense_58/MatMul/ReadVariableOp)^model_14/dense_59/BiasAdd/ReadVariableOp(^model_14/dense_59/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2T
(model_14/dense_56/BiasAdd/ReadVariableOp(model_14/dense_56/BiasAdd/ReadVariableOp2R
'model_14/dense_56/MatMul/ReadVariableOp'model_14/dense_56/MatMul/ReadVariableOp2T
(model_14/dense_57/BiasAdd/ReadVariableOp(model_14/dense_57/BiasAdd/ReadVariableOp2R
'model_14/dense_57/MatMul/ReadVariableOp'model_14/dense_57/MatMul/ReadVariableOp2T
(model_14/dense_58/BiasAdd/ReadVariableOp(model_14/dense_58/BiasAdd/ReadVariableOp2R
'model_14/dense_58/MatMul/ReadVariableOp'model_14/dense_58/MatMul/ReadVariableOp2T
(model_14/dense_59/BiasAdd/ReadVariableOp(model_14/dense_59/BiasAdd/ReadVariableOp2R
'model_14/dense_59/MatMul/ReadVariableOp'model_14/dense_59/MatMul/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_15
Å

)__inference_dense_59_layer_call_fn_268867

inputs
unknown:	ô

	unknown_0:

identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_268252o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
½
³
__inference_loss_fn_1_268906N
:dense_57_kernel_regularizer_square_readvariableop_resource:
ôô
identity¢1dense_57/kernel/Regularizer/Square/ReadVariableOp®
1dense_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_57_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
"dense_57/kernel/Regularizer/SquareSquare9dense_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum&dense_57/kernel/Regularizer/Square:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_57/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_57/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_57/kernel/Regularizer/Square/ReadVariableOp1dense_57/kernel/Regularizer/Square/ReadVariableOp
Ð	
Ä
)__inference_model_14_layer_call_fn_268302
input_15
unknown:
ô
	unknown_0:	ô
	unknown_1:
ôô
	unknown_2:	ô
	unknown_3:
ôô
	unknown_4:	ô
	unknown_5:	ô

	unknown_6:

identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinput_15unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_14_layer_call_and_return_conditional_losses_268283o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_15
½
³
__inference_loss_fn_0_268895N
:dense_56_kernel_regularizer_square_readvariableop_resource:
ô
identity¢1dense_56/kernel/Regularizer/Square/ReadVariableOp®
1dense_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_56_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
ô*
dtype0
"dense_56/kernel/Regularizer/SquareSquare9dense_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôr
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum&dense_56/kernel/Regularizer/Square:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_56/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_56/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_56/kernel/Regularizer/Square/ReadVariableOp1dense_56/kernel/Regularizer/Square/ReadVariableOp
7
Ñ
D__inference_model_14_layer_call_and_return_conditional_losses_268283

inputs#
dense_56_268184:
ô
dense_56_268186:	ô#
dense_57_268207:
ôô
dense_57_268209:	ô#
dense_58_268230:
ôô
dense_58_268232:	ô"
dense_59_268253:	ô

dense_59_268255:

identity¢ dense_56/StatefulPartitionedCall¢1dense_56/kernel/Regularizer/Square/ReadVariableOp¢ dense_57/StatefulPartitionedCall¢1dense_57/kernel/Regularizer/Square/ReadVariableOp¢ dense_58/StatefulPartitionedCall¢1dense_58/kernel/Regularizer/Square/ReadVariableOp¢ dense_59/StatefulPartitionedCall¢1dense_59/kernel/Regularizer/Square/ReadVariableOpñ
 dense_56/StatefulPartitionedCallStatefulPartitionedCallinputsdense_56_268184dense_56_268186*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_268183
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_268207dense_57_268209*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_268206
 dense_58/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0dense_58_268230dense_58_268232*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_268229
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_268253dense_59_268255*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_268252
1dense_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_56_268184* 
_output_shapes
:
ô*
dtype0
"dense_56/kernel/Regularizer/SquareSquare9dense_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôr
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum&dense_56/kernel/Regularizer/Square:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_57_268207* 
_output_shapes
:
ôô*
dtype0
"dense_57/kernel/Regularizer/SquareSquare9dense_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum&dense_57/kernel/Regularizer/Square:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_58_268230* 
_output_shapes
:
ôô*
dtype0
"dense_58/kernel/Regularizer/SquareSquare9dense_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum&dense_58/kernel/Regularizer/Square:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_59_268253*
_output_shapes
:	ô
*
dtype0
"dense_59/kernel/Regularizer/SquareSquare9dense_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô
r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum&dense_59/kernel/Regularizer/Square:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
NoOpNoOp!^dense_56/StatefulPartitionedCall2^dense_56/kernel/Regularizer/Square/ReadVariableOp!^dense_57/StatefulPartitionedCall2^dense_57/kernel/Regularizer/Square/ReadVariableOp!^dense_58/StatefulPartitionedCall2^dense_58/kernel/Regularizer/Square/ReadVariableOp!^dense_59/StatefulPartitionedCall2^dense_59/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2f
1dense_56/kernel/Regularizer/Square/ReadVariableOp1dense_56/kernel/Regularizer/Square/ReadVariableOp2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2f
1dense_57/kernel/Regularizer/Square/ReadVariableOp1dense_57/kernel/Regularizer/Square/ReadVariableOp2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2f
1dense_58/kernel/Regularizer/Square/ReadVariableOp1dense_58/kernel/Regularizer/Square/ReadVariableOp2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2f
1dense_59/kernel/Regularizer/Square/ReadVariableOp1dense_59/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
7
Ó
D__inference_model_14_layer_call_and_return_conditional_losses_268501
input_15#
dense_56_268456:
ô
dense_56_268458:	ô#
dense_57_268461:
ôô
dense_57_268463:	ô#
dense_58_268466:
ôô
dense_58_268468:	ô"
dense_59_268471:	ô

dense_59_268473:

identity¢ dense_56/StatefulPartitionedCall¢1dense_56/kernel/Regularizer/Square/ReadVariableOp¢ dense_57/StatefulPartitionedCall¢1dense_57/kernel/Regularizer/Square/ReadVariableOp¢ dense_58/StatefulPartitionedCall¢1dense_58/kernel/Regularizer/Square/ReadVariableOp¢ dense_59/StatefulPartitionedCall¢1dense_59/kernel/Regularizer/Square/ReadVariableOpó
 dense_56/StatefulPartitionedCallStatefulPartitionedCallinput_15dense_56_268456dense_56_268458*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_268183
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_268461dense_57_268463*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_268206
 dense_58/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0dense_58_268466dense_58_268468*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_268229
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_268471dense_59_268473*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_268252
1dense_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_56_268456* 
_output_shapes
:
ô*
dtype0
"dense_56/kernel/Regularizer/SquareSquare9dense_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôr
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum&dense_56/kernel/Regularizer/Square:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_57_268461* 
_output_shapes
:
ôô*
dtype0
"dense_57/kernel/Regularizer/SquareSquare9dense_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum&dense_57/kernel/Regularizer/Square:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_58_268466* 
_output_shapes
:
ôô*
dtype0
"dense_58/kernel/Regularizer/SquareSquare9dense_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum&dense_58/kernel/Regularizer/Square:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_59_268471*
_output_shapes
:	ô
*
dtype0
"dense_59/kernel/Regularizer/SquareSquare9dense_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô
r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum&dense_59/kernel/Regularizer/Square:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
NoOpNoOp!^dense_56/StatefulPartitionedCall2^dense_56/kernel/Regularizer/Square/ReadVariableOp!^dense_57/StatefulPartitionedCall2^dense_57/kernel/Regularizer/Square/ReadVariableOp!^dense_58/StatefulPartitionedCall2^dense_58/kernel/Regularizer/Square/ReadVariableOp!^dense_59/StatefulPartitionedCall2^dense_59/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2f
1dense_56/kernel/Regularizer/Square/ReadVariableOp1dense_56/kernel/Regularizer/Square/ReadVariableOp2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2f
1dense_57/kernel/Regularizer/Square/ReadVariableOp1dense_57/kernel/Regularizer/Square/ReadVariableOp2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2f
1dense_58/kernel/Regularizer/Square/ReadVariableOp1dense_58/kernel/Regularizer/Square/ReadVariableOp2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2f
1dense_59/kernel/Regularizer/Square/ReadVariableOp1dense_59/kernel/Regularizer/Square/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_15
7
Ó
D__inference_model_14_layer_call_and_return_conditional_losses_268549
input_15#
dense_56_268504:
ô
dense_56_268506:	ô#
dense_57_268509:
ôô
dense_57_268511:	ô#
dense_58_268514:
ôô
dense_58_268516:	ô"
dense_59_268519:	ô

dense_59_268521:

identity¢ dense_56/StatefulPartitionedCall¢1dense_56/kernel/Regularizer/Square/ReadVariableOp¢ dense_57/StatefulPartitionedCall¢1dense_57/kernel/Regularizer/Square/ReadVariableOp¢ dense_58/StatefulPartitionedCall¢1dense_58/kernel/Regularizer/Square/ReadVariableOp¢ dense_59/StatefulPartitionedCall¢1dense_59/kernel/Regularizer/Square/ReadVariableOpó
 dense_56/StatefulPartitionedCallStatefulPartitionedCallinput_15dense_56_268504dense_56_268506*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_268183
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_268509dense_57_268511*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_268206
 dense_58/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0dense_58_268514dense_58_268516*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_268229
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_268519dense_59_268521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_268252
1dense_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_56_268504* 
_output_shapes
:
ô*
dtype0
"dense_56/kernel/Regularizer/SquareSquare9dense_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôr
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum&dense_56/kernel/Regularizer/Square:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_57_268509* 
_output_shapes
:
ôô*
dtype0
"dense_57/kernel/Regularizer/SquareSquare9dense_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum&dense_57/kernel/Regularizer/Square:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_58_268514* 
_output_shapes
:
ôô*
dtype0
"dense_58/kernel/Regularizer/SquareSquare9dense_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum&dense_58/kernel/Regularizer/Square:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_59_268519*
_output_shapes
:	ô
*
dtype0
"dense_59/kernel/Regularizer/SquareSquare9dense_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô
r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum&dense_59/kernel/Regularizer/Square:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
NoOpNoOp!^dense_56/StatefulPartitionedCall2^dense_56/kernel/Regularizer/Square/ReadVariableOp!^dense_57/StatefulPartitionedCall2^dense_57/kernel/Regularizer/Square/ReadVariableOp!^dense_58/StatefulPartitionedCall2^dense_58/kernel/Regularizer/Square/ReadVariableOp!^dense_59/StatefulPartitionedCall2^dense_59/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2f
1dense_56/kernel/Regularizer/Square/ReadVariableOp1dense_56/kernel/Regularizer/Square/ReadVariableOp2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2f
1dense_57/kernel/Regularizer/Square/ReadVariableOp1dense_57/kernel/Regularizer/Square/ReadVariableOp2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2f
1dense_58/kernel/Regularizer/Square/ReadVariableOp1dense_58/kernel/Regularizer/Square/ReadVariableOp2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2f
1dense_59/kernel/Regularizer/Square/ReadVariableOp1dense_59/kernel/Regularizer/Square/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_15
7
Ñ
D__inference_model_14_layer_call_and_return_conditional_losses_268413

inputs#
dense_56_268368:
ô
dense_56_268370:	ô#
dense_57_268373:
ôô
dense_57_268375:	ô#
dense_58_268378:
ôô
dense_58_268380:	ô"
dense_59_268383:	ô

dense_59_268385:

identity¢ dense_56/StatefulPartitionedCall¢1dense_56/kernel/Regularizer/Square/ReadVariableOp¢ dense_57/StatefulPartitionedCall¢1dense_57/kernel/Regularizer/Square/ReadVariableOp¢ dense_58/StatefulPartitionedCall¢1dense_58/kernel/Regularizer/Square/ReadVariableOp¢ dense_59/StatefulPartitionedCall¢1dense_59/kernel/Regularizer/Square/ReadVariableOpñ
 dense_56/StatefulPartitionedCallStatefulPartitionedCallinputsdense_56_268368dense_56_268370*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_268183
 dense_57/StatefulPartitionedCallStatefulPartitionedCall)dense_56/StatefulPartitionedCall:output:0dense_57_268373dense_57_268375*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_57_layer_call_and_return_conditional_losses_268206
 dense_58/StatefulPartitionedCallStatefulPartitionedCall)dense_57/StatefulPartitionedCall:output:0dense_58_268378dense_58_268380*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_58_layer_call_and_return_conditional_losses_268229
 dense_59/StatefulPartitionedCallStatefulPartitionedCall)dense_58/StatefulPartitionedCall:output:0dense_59_268383dense_59_268385*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_59_layer_call_and_return_conditional_losses_268252
1dense_56/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_56_268368* 
_output_shapes
:
ô*
dtype0
"dense_56/kernel/Regularizer/SquareSquare9dense_56/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôr
!dense_56/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_56/kernel/Regularizer/SumSum&dense_56/kernel/Regularizer/Square:y:0*dense_56/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_56/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_56/kernel/Regularizer/mulMul*dense_56/kernel/Regularizer/mul/x:output:0(dense_56/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_57/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_57_268373* 
_output_shapes
:
ôô*
dtype0
"dense_57/kernel/Regularizer/SquareSquare9dense_57/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_57/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_57/kernel/Regularizer/SumSum&dense_57/kernel/Regularizer/Square:y:0*dense_57/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_57/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_57/kernel/Regularizer/mulMul*dense_57/kernel/Regularizer/mul/x:output:0(dense_57/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_58_268378* 
_output_shapes
:
ôô*
dtype0
"dense_58/kernel/Regularizer/SquareSquare9dense_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum&dense_58/kernel/Regularizer/Square:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
1dense_59/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_59_268383*
_output_shapes
:	ô
*
dtype0
"dense_59/kernel/Regularizer/SquareSquare9dense_59/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	ô
r
!dense_59/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_59/kernel/Regularizer/SumSum&dense_59/kernel/Regularizer/Square:y:0*dense_59/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_59/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o:
dense_59/kernel/Regularizer/mulMul*dense_59/kernel/Regularizer/mul/x:output:0(dense_59/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_59/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
NoOpNoOp!^dense_56/StatefulPartitionedCall2^dense_56/kernel/Regularizer/Square/ReadVariableOp!^dense_57/StatefulPartitionedCall2^dense_57/kernel/Regularizer/Square/ReadVariableOp!^dense_58/StatefulPartitionedCall2^dense_58/kernel/Regularizer/Square/ReadVariableOp!^dense_59/StatefulPartitionedCall2^dense_59/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall2f
1dense_56/kernel/Regularizer/Square/ReadVariableOp1dense_56/kernel/Regularizer/Square/ReadVariableOp2D
 dense_57/StatefulPartitionedCall dense_57/StatefulPartitionedCall2f
1dense_57/kernel/Regularizer/Square/ReadVariableOp1dense_57/kernel/Regularizer/Square/ReadVariableOp2D
 dense_58/StatefulPartitionedCall dense_58/StatefulPartitionedCall2f
1dense_58/kernel/Regularizer/Square/ReadVariableOp1dense_58/kernel/Regularizer/Square/ReadVariableOp2D
 dense_59/StatefulPartitionedCall dense_59/StatefulPartitionedCall2f
1dense_59/kernel/Regularizer/Square/ReadVariableOp1dense_59/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

)__inference_dense_56_layer_call_fn_268771

inputs
unknown:
ô
	unknown_0:	ô
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_56_layer_call_and_return_conditional_losses_268183p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
³
__inference_loss_fn_2_268917N
:dense_58_kernel_regularizer_square_readvariableop_resource:
ôô
identity¢1dense_58/kernel/Regularizer/Square/ReadVariableOp®
1dense_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_58_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
"dense_58/kernel/Regularizer/SquareSquare9dense_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum&dense_58/kernel/Regularizer/Square:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentity#dense_58/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: z
NoOpNoOp2^dense_58/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1dense_58/kernel/Regularizer/Square/ReadVariableOp1dense_58/kernel/Regularizer/Square/ReadVariableOp
¼
¬
D__inference_dense_58_layer_call_and_return_conditional_losses_268852

inputs2
matmul_readvariableop_resource:
ôô.
biasadd_readvariableop_resource:	ô
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢1dense_58/kernel/Regularizer/Square/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ô*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿôQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
1dense_58/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôô*
dtype0
"dense_58/kernel/Regularizer/SquareSquare9dense_58/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
ôôr
!dense_58/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
dense_58/kernel/Regularizer/SumSum&dense_58/kernel/Regularizer/Square:y:0*dense_58/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: f
!dense_58/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
dense_58/kernel/Regularizer/mulMul*dense_58/kernel/Regularizer/mul/x:output:0(dense_58/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô«
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_58/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_58/kernel/Regularizer/Square/ReadVariableOp1dense_58/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*®
serving_default
>
input_152
serving_default_input_15:0ÿÿÿÿÿÿÿÿÿ<
dense_590
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:f

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
»

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
ã

/beta_1

0beta_2
	1decay
2learning_rate
3iterm]m^m_m`ma mb'mc(mdvevfvgvhvi vj'vk(vl"
	optimizer
X
0
1
2
3
4
 5
'6
(7"
trackable_list_wrapper
X
0
1
2
3
4
 5
'6
(7"
trackable_list_wrapper
<
40
51
62
73"
trackable_list_wrapper
Ê
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò2ï
)__inference_model_14_layer_call_fn_268302
)__inference_model_14_layer_call_fn_268600
)__inference_model_14_layer_call_fn_268621
)__inference_model_14_layer_call_fn_268453À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_model_14_layer_call_and_return_conditional_losses_268677
D__inference_model_14_layer_call_and_return_conditional_losses_268733
D__inference_model_14_layer_call_and_return_conditional_losses_268501
D__inference_model_14_layer_call_and_return_conditional_losses_268549À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÍBÊ
!__inference__wrapped_model_268159input_15"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
=serving_default"
signature_map
#:!
ô2dense_56/kernel
:ô2dense_56/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
40"
trackable_list_wrapper
­
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_56_layer_call_fn_268771¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_56_layer_call_and_return_conditional_losses_268788¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!
ôô2dense_57/kernel
:ô2dense_57/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
50"
trackable_list_wrapper
­
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_57_layer_call_fn_268803¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_57_layer_call_and_return_conditional_losses_268820¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!
ôô2dense_58/kernel
:ô2dense_58/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
'
60"
trackable_list_wrapper
­
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_58_layer_call_fn_268835¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_58_layer_call_and_return_conditional_losses_268852¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 	ô
2dense_59/kernel
:
2dense_59/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
'
70"
trackable_list_wrapper
­
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_59_layer_call_fn_268867¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_59_layer_call_and_return_conditional_losses_268884¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
³2°
__inference_loss_fn_0_268895
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_1_268906
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_2_268917
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_3_268928
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÌBÉ
$__inference_signature_wrapper_268756input_15"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
60"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
70"
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ttotal
	Ucount
V	variables
W	keras_api"
_tf_keras_metric
^
	Xtotal
	Ycount
Z
_fn_kwargs
[	variables
\	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
T0
U1"
trackable_list_wrapper
-
V	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
X0
Y1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
(:&
ô2Adam/dense_56/kernel/m
!:ô2Adam/dense_56/bias/m
(:&
ôô2Adam/dense_57/kernel/m
!:ô2Adam/dense_57/bias/m
(:&
ôô2Adam/dense_58/kernel/m
!:ô2Adam/dense_58/bias/m
':%	ô
2Adam/dense_59/kernel/m
 :
2Adam/dense_59/bias/m
(:&
ô2Adam/dense_56/kernel/v
!:ô2Adam/dense_56/bias/v
(:&
ôô2Adam/dense_57/kernel/v
!:ô2Adam/dense_57/bias/v
(:&
ôô2Adam/dense_58/kernel/v
!:ô2Adam/dense_58/bias/v
':%	ô
2Adam/dense_59/kernel/v
 :
2Adam/dense_59/bias/v
!__inference__wrapped_model_268159s '(2¢/
(¢%
# 
input_15ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_59"
dense_59ÿÿÿÿÿÿÿÿÿ
¦
D__inference_dense_56_layer_call_and_return_conditional_losses_268788^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿô
 ~
)__inference_dense_56_layer_call_fn_268771Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿô¦
D__inference_dense_57_layer_call_and_return_conditional_losses_268820^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿô
ª "&¢#

0ÿÿÿÿÿÿÿÿÿô
 ~
)__inference_dense_57_layer_call_fn_268803Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿô
ª "ÿÿÿÿÿÿÿÿÿô¦
D__inference_dense_58_layer_call_and_return_conditional_losses_268852^ 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿô
ª "&¢#

0ÿÿÿÿÿÿÿÿÿô
 ~
)__inference_dense_58_layer_call_fn_268835Q 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿô
ª "ÿÿÿÿÿÿÿÿÿô¥
D__inference_dense_59_layer_call_and_return_conditional_losses_268884]'(0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿô
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 }
)__inference_dense_59_layer_call_fn_268867P'(0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿô
ª "ÿÿÿÿÿÿÿÿÿ
;
__inference_loss_fn_0_268895¢

¢ 
ª " ;
__inference_loss_fn_1_268906¢

¢ 
ª " ;
__inference_loss_fn_2_268917¢

¢ 
ª " ;
__inference_loss_fn_3_268928'¢

¢ 
ª " µ
D__inference_model_14_layer_call_and_return_conditional_losses_268501m '(:¢7
0¢-
# 
input_15ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 µ
D__inference_model_14_layer_call_and_return_conditional_losses_268549m '(:¢7
0¢-
# 
input_15ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ³
D__inference_model_14_layer_call_and_return_conditional_losses_268677k '(8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ³
D__inference_model_14_layer_call_and_return_conditional_losses_268733k '(8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
)__inference_model_14_layer_call_fn_268302` '(:¢7
0¢-
# 
input_15ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

)__inference_model_14_layer_call_fn_268453` '(:¢7
0¢-
# 
input_15ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

)__inference_model_14_layer_call_fn_268600^ '(8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

)__inference_model_14_layer_call_fn_268621^ '(8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
§
$__inference_signature_wrapper_268756 '(>¢;
¢ 
4ª1
/
input_15# 
input_15ÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_59"
dense_59ÿÿÿÿÿÿÿÿÿ
