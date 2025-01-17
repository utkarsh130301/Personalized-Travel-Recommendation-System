from common import f1_scores
import statistics

## output recall precision scores
#SPMFPredictor.py
## calculate F1 scores / precision / recall scores
def scores(city, algo):
    print("CITY ==> ", city)
    file=f"spmf_{city}_{algo}.txt"
    rscores=[]
    pscores=[]
    fscores=[]
    with open(file) as f:
        line = f.readline()
        while line:
            line = f.readline().strip()
            tab=line.split(" ")
            if city in tab[0] and len(tab) >= 5  :
                tryj,pred = tab[2], tab[4]
                algo=str(tab[0])[len(city)+2:].replace(">","")
                (precision,recall,f1score) = f1_scores(tryj,pred)
                rscores.append(recall)
                pscores.append(precision)
                fscores.append(f1score)
                #print(f"> {city}, {algo}, prec:{precision}, recall:{recall}, f1:{f1score} <= {tryj}, {pred}"  )
    if len(rscores)>1 and len(pscores)>1 and len(fscores)>1:
        mean_rscores = 100 * statistics.mean(rscores)
        mean_pscores = 100 * statistics.mean(pscores)
        mean_f1scores = 100 * statistics.mean(fscores)

        hmean_rscores = 100 * statistics.harmonic_mean(rscores)
        hmean_pscores = 100 * statistics.harmonic_mean(pscores)
        hmean_f1scores = 100 * statistics.harmonic_mean(fscores)
        
        sd_rscores = 100 * statistics.mean(rscores)
        sd_pscores = 100 * statistics.mean(rscores)
        sd_f1scores = 100 * statistics.mean(rscores)
        
        print("city:{} algo:{} re_score => mean:{} hmean:{} stdev:{}".format(city, algo, mean_rscores, hmean_rscores, sd_rscores))
        print("city:{} algo:{} pr_score => mean:{} hmean:{} stdev:{}".format(city, algo, mean_pscores, hmean_pscores, sd_pscores))
        print("city:{} algo:{} f1_score => mean:{} hmean:{} stdev:{}".format(city, algo, mean_f1scores,hmean_f1scores,sd_f1scores))

def main():
    # read in from  spmf.sh / 
    for city in ['Buda','Delh','Edin','Glas','Osak','Pert','Toro']:
        for algo in ['DG','CPT','CPTPlus','TDAG','LZ78','MarkovAllK','MarkovFirstOrder' ]:
            scores(city, algo)

if __name__ == '__main__':
  main()
