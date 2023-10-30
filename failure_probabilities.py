def p_ab(p_ba:float, pa:float, pb:float)->float:
    """P(A|B) = P(B|A)*P(A)/P(B)

    Args:
        p_ba (float): P(B|A)
        pa (float): P(A)
        pb (float): P(A)

    Returns:
        float: P(A|B)
    """
    return p_ba*pa/pb

vcf = 0.05 #Vehicle component failure probability
sf = 0.04 #Software failure
vca = 0.98 #Vehicle control algorithm failure

scdi = 0.95 # Special conditions of driving infrastructure
iaboru = 0.5 #Inadequate action by other road users
icd = 0.8 #Inadequate car drivers

print("Probability of Vehicle Control Algorithm AND Inadequate Car Driver failure", end=" :") 
vca *= sf*vcf
icd *= iaboru*scdi
print(vca+icd)

print("Probability of Software Failure AND Extreme Weather Conditions", end=" :") 
ewc = 0.3
sfAewc = vcf * sf * scdi * ewc
print(sfAewc)