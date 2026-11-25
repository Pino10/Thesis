"""Figure 4.1 of the revised thesis: unseen-hostname recall against parameter count.
Values are the three-seed means from zero_day_and_metrics.csv and pruning_protected_iterative.csv.

    python code/revision/figure41.py
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif","Times New Roman"],"font.size":11})
pts=[("S-LIN",25,98.71,"D",True,(15,-2)),("S-16",545,99.53,"*",True,(12,8)),("S-64",3713,99.57,"^",True,(-30,10)),
     ("Pruned 90%\n(output layer protected)",10421,99.65,"s",True,(12,6)),("S-128",13569,99.54,"^",False,(-20,-22)),
     ("Baseline U\n(teacher)",96385,99.53,"o",False,(-25,-32))]
fig,ax=plt.subplots(figsize=(8,4.9),dpi=200)
front=[p for p in pts if p[4]]
ax.plot([p[1] for p in front],[p[2] for p in front],"--",color="0.45",lw=1)
for n,x,y,m,f,off in pts:
    ax.plot(x,y,m,ms=13 if m=="*" else 9,mfc="black" if f else "white",mec="black",mew=1.4)
    ax.annotate(n,(x,y),xytext=off,textcoords="offset points",fontsize=11,ha="left" if off[0]>0 else "center")
ax.set_xscale("log"); ax.set_xlim(15,3e5); ax.set_ylim(98.4,99.95)
ax.set_xlabel("Model size (parameters, log scale)"); ax.set_ylabel("Unseen-domain zero-day recall (%)")
ax.grid(True,which="major",ls=":",color="0.8")
ax.text(0.98,0.05,"Filled markers: non-dominated configurations",transform=ax.transAxes,ha="right",fontsize=10)
fig.tight_layout(); fig.savefig(str(Path(__file__).resolve().parents[2] / "results" / "revision" / "figure41.png"),dpi=200)
