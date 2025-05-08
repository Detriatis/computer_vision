import gc, torch, numpy as np
from sklearn.model_selection import StratifiedKFold
from main.utils.scikit_interface import SKlearnPyTorchClassifier
from main.pipeline.cnn import Net

# --- create a *tiny* dummy set so the loop is fast -------------
X_small = np.random.rand(120, 96, 96, 1).astype("float32")
y_small = np.random.randint(0, 10, size=120)

clf = SKlearnPyTorchClassifier(Net, epochs=4, batch_size=32, device="cuda")

def one_cv_run():
    # 3‑fold CV but only ONE candidate → 3 fits
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    for train, val in cv.split(X_small, y_small):
        clone = clf.__class__(**clf.get_params(deep=True))
        clone.fit(X_small[train], y_small[train])
        _ = clone.score(X_small[val], y_small[val])
        # move weights off GPU & free memory
        if hasattr(clone, "model"):
            clone.model.to("cpu")
        del clone
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

# --- hammer it 100× (≈ 300 fits) -------------------------------
for i in range(100):
    one_cv_run()
print("finished 100 cycles without GPU fault ✔︎")
