# 10-605:HW#7 DSGD-MF on Spark

Uses PySpark to run DSGD-MF. Written with the Netflix dataset in mind.

To run the code, execute this command:

spark-submit dsgd_mf.py <num_factors> <num_workers> <num_iterations> <beta_value> <lambda_value> \
	     <inputV_filepath> <outputW_filepath> <outputH_filepath>

Num_factors: The number of factors in the factor matrices.
Num_workers: The number of Spark workers.
Num_iterations: The number of iterations to run for.
Beta_value: The value of beta. Governs the speed at which the learning rate decays.
Lambda_value: The regularization coefficient.
InputV_filepath: File path for the csv or directory of Netflix files representing V.
OutputW_filepath: Filename for the csv representing the resulting W matrix.
OutputH_filepath: Filename for the csv representing the resulting H matrix.

Note to 10-605 graders: dsgd_mf.py implements DSGD-MF according the homework spec. dsgd_mf_2.py only
updates the learning rate at the end of a pass through the data. I used the latter for the evaluation (reconstruction error, speed, memory efficiency) and the former to generate the experimental data. Yipei Wang said this was all right in office hours (4/15).