import numpy as np


class met_dataclass(object):
    def __init__(self, dat):
        self.XT = dat['XT']
        self.XE = dat['XE']
        self.XM = dat['XM']
        self.Xsd = dat['Xsd']
        self.cluster_label = dat['cluster_label']
        self.cluster_id = dat['cluster_id']
        self.cluster_color = dat['cluster_color']
        self.specimen_id = dat['specimen_id']
        self.gene_ids = dat['gene_ids']
        self.E_features = dat['E_features']
        self.norm2px = dat['norm2px']

    def __getitem__(self, inds):
        # convert a simple indsex x[y] to a tuple for consistency
        if not isinstance(inds, tuple):
            inds = tuple(inds)
        return met_dataclass(dict(XT=self.XT[inds[0]],
                                  XE=self.XE[inds[0]],
                                  XM=self.XM[inds[0]],
                                  Xsd=self.Xsd[inds[0]],
                                  cluster_label=self.cluster_label[inds[0]],
                                  cluster_id=self.cluster_id[inds[0]],
                                  cluster_color=self.cluster_color[inds[0]],
                                  specimen_id=self.specimen_id[inds[0]],
                                  gene_ids=self.gene_ids,
                                  E_features=self.E_features,
                                  norm2px=self.norm2px))


    def __repr__(self):
        return f'met data with {self.XT.shape[0]} cells'

    @staticmethod
    def valid_data(x):
        return np.any(~np.isnan(x).reshape(x.shape[0], -1), axis=1)

    @property
    def isT_1d(self): return self.valid_data(self.XT)

    @property
    def isE_1d(self): return self.valid_data(self.XE)

    @property
    def isM_1d(self): return self.valid_data(self.XM)

    @property
    def isT_ind(self): return np.flatnonzero(self.isT_1d)

    @property
    def isE_ind(self): return np.flatnonzero(self.isE_1d)

    @property
    def isM_ind(self): return np.flatnonzero(self.isM_1d)

    @property
    def Xsd_px(self): return self.Xsd * self.norm2px

    @property
    def XM_centered(self):
        return self.soma_center(XM=self.XM, Xsd=self.Xsd, norm2px=self.norm2px, jitter_frac=0)

    def summary(self):
        print(f"T shape {self.XT.shape}")
        print(f"E shape {self.XE.shape}")
        print(f"M shape {self.XM.shape}")
        print(f"sd shape {self.Xsd.shape}")

        # find all-nan samples boolean
        def allnans(x): return np.all(
            np.isnan(x.reshape(np.shape(x)[0], -1)), axis=(1))

        m_T = ~allnans(self.XT)
        m_E = ~allnans(self.XE)
        m_M = ~allnans(self.XM)

        print('\nSamples with at least one non-nan features')
        print(f'{np.sum(m_T)} cells in T')
        print(f'{np.sum(m_E)} cells in E')
        print(f'{np.sum(m_M)} cells in M')

        def paired(x, y): return np.sum(np.logical_and(x, y))
        print('\nPaired samples, allowing for nans in a strict subset of features in both modalities')
        print(f'{paired(m_T,m_E)} cells paired in T and E')
        print(f'{paired(m_T,m_M)} cells paired in T and M')
        print(f'{paired(m_E,m_M)} cells paired in E and M')
        return

    @staticmethod
    def soma_center(XM, Xsd, norm2px, jitter_frac=0):
        # nans are treated as zeros
        assert (np.nanmax(Xsd) <= 1) and (np.nanmin(Xsd) >= 0), 'Xsd expected in range (0,1)'
        jitter = (np.random.random(Xsd.shape) - 0.5)*jitter_frac
        Xsd_jitter = np.clip(Xsd + jitter, 0, 1)

        Xsd_px = np.round(Xsd_jitter * norm2px)
        Xsd_px = np.nan_to_num(Xsd_px).astype(int)

        # padded density map is double the size of the original in H dimension
        pad = XM.shape[1] // 2
        XM_centered = np.zeros(np.array(XM.shape) + np.array((0, 2*pad, 0, 0)))
        new_zero_px = XM_centered.shape[1] // 2 - Xsd_px
        for i in range(XM_centered.shape[0]):
            XM_centered[i, new_zero_px[i]:new_zero_px[i] +
                        XM.shape[1], :, :] = XM[i, ...]

        setnan = np.apply_over_axes(np.all, np.isnan(XM), axes=[1, 2])
        setnan = np.broadcast_to(setnan, XM_centered.shape)
        XM_centered[setnan] = np.nan
        return XM_centered
