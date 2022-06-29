
import PastaQ: gate
using HDF5

#####################################
##    Projective Measurements      ##
#####################################
gate(::GateName"Π0") =
  [1 0
   0 0]
gate(::GateName"Π1") =
  [0 0
   0 1]
# new 2 qubit projectors
gate(::GateName"Π00") =
[1 0 0 0
 0 0 0 0
 0 0 0 0
 0 0 0 0]
gate(::GateName"Π10") =
[0 0 0 0
 0 1 0 0
 0 0 0 0
 0 0 0 0]
gate(::GateName"Π01") =
[0 0 0 0
 0 0 0 0
 0 0 1 0
 0 0 0 0]
gate(::GateName"Π11") =
[0 0 0 0
 0 0 0 0
 0 0 0 0
 0 0 0 1]

#####################################################################################################
## Gates for completely packed loop model with crossings (CPLC) from Jordan–Wigner Transformations ##
#####################################################################################################
 gate(::GateName"Xup") =
 1/2*([1 1
       1 1])
 gate(::GateName"Xdown") =
 1/2*([1 -1
      -1 1])

gate(::GateName"ΠJWCPLC_UOdd") =
  1/sqrt(2)*([1 0
              0 1] + im *[0 1
                          1 0])

gate(::GateName"ΠJWCPLC_UOdd1") =
[1 0
 0 1]

gate(::GateName"ΠJWCPLC_UEven") =
    1/sqrt(2)*(
    [1 0 0 0
     0 1 0 0
     0 0 1 0
     0 0 0 1] + im * [1 0 0 0
                      0 -1 0 0
                      0 0 -1 0
                      0 0 0  1])

gate(::GateName"ΠJWCPLC_UEven1") =
[1 0 0 0
 0 1 0 0
 0 0 1 0
 0 0 0 1]

gate(::GateName"ZOp") =
 [1 0
  0 -1]

gate(::GateName"ZZup") =
[1 0 0 0
 0 0 0 0
 0 0 0 0
 0 0 0 1]

 gate(::GateName"ZZdown") =
 [0 0 0 0
  0 1 0 0
  0 0 1 0
  0 0 0 0]

#####################################
## Loading Saved Clifford Gate Set ##
#####################################
fid = h5open(string(@__DIR__, "/in_data/clifford_dict.h5"), "r")
clifford_samples = 99999
clifford_dict = Dict()
for c in 1:1:clifford_samples
    dataset_name = "clifford_$c"
    obj = fid[dataset_name]
    read_obj = read(obj)
    this_clifford = [
    read_obj[(1,1)...]+read_obj[(2,1)...]im read_obj[(1,2)...]+read_obj[(2,2)...]im read_obj[(1,3)...]+read_obj[(2,3)...]im read_obj[(1,4)...]+read_obj[(2,4)...]im
    read_obj[(1,5)...]+read_obj[(2,5)...]im read_obj[(1,6)...]+read_obj[(2,6)...]im read_obj[(1,7)...]+read_obj[(2,7)...]im read_obj[(1,8)...]+read_obj[(2,8)...]im
    read_obj[(1,9)...]+read_obj[(2,9)...]im read_obj[(1,10)...]+read_obj[(2,10)...]im read_obj[(1,11)...]+read_obj[(2,11)...]im read_obj[(1,12)...]+read_obj[(2,12)...]im
    read_obj[(1,13)...]+read_obj[(2,13)...]im read_obj[(1,14)...]+read_obj[(2,14)...]im read_obj[(1,15)...]+read_obj[(2,15)...]im read_obj[(1,16)...]+read_obj[(2,16)...]im
    ]
    clifford_dict["$c"] = this_clifford
end
