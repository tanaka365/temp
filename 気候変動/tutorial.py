# %%
import numpy as np
import pandas as pd

from climada.engine import Impact
from climada.entity import ImpactFuncSet, ImpfTropCyclone, LitPop
from climada.hazard import Centroids, TCTracks, TropCyclone

# %%
# exposureの設定
exp_jp = LitPop.from_countries(countries=["JP"], fin_mode="income_group")
exp_jp.check()

# %%
# exposureの可視化
exp_jp.plot_raster()
print("\n Raster properties exposures:", exp_jp.meta)


# %%
# hazardの設定
# http://agora.ex.nii.ac.jp/digital-typhoon/ibtracs/
# 上記のサイトから日本の台風のstorm_idを取得可能
storm_id = "2018229N11160"
tr_japan = TCTracks.from_ibtracs_netcdf(provider="usa", storm_id=storm_id)
ax = tr_japan.plot()
ax.set_title("JAPAN")

# %%
# ストームの情報
tr_japan.get_track(storm_id)

# %%
# 有効ランダムウォークに基づいて、各トラックの合成トラックを生成することが可能。
# calc_perturbed_trajectoriesは"nb_synth_tracks"個の合成トラックのアンサンブルを生成。
tr_japan.equal_timestep()
tr_japan.calc_perturbed_trajectories(nb_synth_tracks=1)
tr_japan.plot()

# %%
# centroidを設定
lat = exp_jp.gdf["latitude"].values
lon = exp_jp.gdf["longitude"].values
centrs = Centroids.from_lat_lon(lat, lon)
centrs.check()
centrs.plot()

# %%
# TropCycloneクラスを用いて、上記セントロイド上のザード強度を算出する。
tc_japan = TropCyclone.from_tracks(tr_japan, centroids=centrs)
tc_japan.check()

# %%
# impact function TC の設定
impf_tc = ImpfTropCyclone.from_emanuel_usa()

# add the impact function to an Impact function set
impf_set = ImpactFuncSet()
impf_set.append(impf_tc)
impf_set.check()

# %%
# Get the hazard type and hazard id
[haz_type] = impf_set.get_hazard_types()
[haz_id] = impf_set.get_ids()[haz_type]
print(f"hazard type: {haz_type}, hazard id: {haz_id}")

# %%
# Exposures: rename column and assign id
exp_jp.gdf.rename(columns={"impf_": "impf_" + haz_type}, inplace=True)
exp_jp.gdf["impf_" + haz_type] = haz_id
exp_jp.check()
exp_jp.gdf.head()

# %%
# impactの計算
# ! 風速の確認
imp = Impact()
imp.calc(
    exp_jp, impf_set, tc_japan, save_mat=True
)  # 　Do not save the results geographically resolved (only aggregate values)

# %%
exp_jp.gdf

# %%
print(f"Aggregated average annual impact: {round(imp.aai_agg,0)} $")

# %%
imp.plot_raster_eai_exposure()

# %%
# Compute exceedance frequency curve
freq_curve = imp.calc_freq_curve()
freq_curve.plot()

# %%

# %%

# %%

# %%
