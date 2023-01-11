import numpy as np

import plotly.express as px

import jpype

import hurwitz


def calc_TE(data : np.array, TE_ver : int = 1, ):
    N = data.shape[1]
    # JIDT jar library path
    jarLocation = "JIDT/infodynamics.jar"
    # Start the JVM
    if not jpype.isJVMStarted():
        jpype.startJVM("/Library/Java/JavaVirtualMachines/jdk-18.0.1.1.jdk/Contents/Home/lib/libjli.dylib", "-ea", "-Djava.class.path=" + jarLocation)

    TE_vals = []

    # Construct the calculator:
    if TE_ver == 1:
        calcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").ConditionalTransferEntropyCalculatorKraskov
    elif TE_ver == 2:
        calcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").ConditionalMutualInfoCalculatorMultiVariateKraskov2
        # calcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorMultiVariateKraskov 
    else:
        print("TE_ver not valid.")       

    calc = calcClass()
    # Set any properties to non-default values:
    # calc.setProperty("AUTO_EMBED_METHOD", "RAGWITZ_DEST_ONLY")
    # calc.setProperty("NUM_THREADS", "USE_ALL")

    third_excluded = {
        "01" : 2,
        "12" : 0,
        "02" : 1
    }

    G_te = np.matrix([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])

    # Compute for all pairs:
    for s in range(N):
        for d in range(N):
            # For each source-dest pair:
            if (s == d):
                continue

            if TE_ver == 1:
                # Initialise the calculator for (re-)use:
                calc.initialise()

                source = data[:,s]
                destination = data[:,d]
                conditional = data[:,third_excluded["".join(sorted([str(s),str(d)]))]]
            elif TE_ver == 2:
                # Initialise the calculator for (re-)use:
                calc.initialise(1,1,2)

                source = [[x] for x in data[:-1,s]]
                destination = [[x] for x in (data[1:,d]-data[:-1,d])]
                conditional = [[data[t,c] for c in [0,1,2] if c != s] for t in range(1,data.shape[0])]

            # Supply the sample data:
            calc.setObservations(source, destination, conditional)
            # Compute the estimate:
            result = calc.computeAverageLocalOfObservations()
            # Compute the (statistical significance via) null distribution empirically (e.g. with 100 permutations):
            measDist = calc.computeSignificance(100)

            pVal = measDist.pValue

            if pVal <= 1.0:
                G_te[d,s] = 1-pVal

            # print("Transfer entropy ({}->{}): {:.4f} nats (std = {:.4f} ± {:.4f}, p < {:.4f}). Correlation = {}".format(s, d, result, measDist.getMeanOfDistribution(), measDist.getStdOfDistribution(), pVal, np.corrcoef(source, destination)[0,1]))
            # TE_vals.append(result)

    # fig = px.imshow(G_te)
    # fig.show()
    return G_te

def compare(G_te : np.matrix, A_mat : np.matrix):
    print(G_te)
    print(A_mat)
    if np.sum(np.abs(np.sign(G_te))-np.abs(np.sign(A_mat))) == 0:
        return True
    return False

if __name__ == "__main__":
    A_mat = hurwitz.gen_hurwitz()
    
    tot_reps = 1
    success = 0
    for rep in range(tot_reps):
        data = hurwitz.run_process(A_mat, 10000)
        G_te = calc_TE(data, TE_ver=1)
        report = "Test #{} was a {}"
        if compare(G_te, A_mat):
            success += 1
            report = report.format(rep,"Success")
        else:
            report = report.format(rep,"Failure")
        print(report)
    print("Success rate: {}/{}".format(success,tot_reps))