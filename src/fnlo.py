import numpy
import fastnlo
from fastnlo import fastNLOLHAPDF, fastNLOAlphas
fastnlo.SetGlobalVerbosity(fastnlo.WARNING)


class fastNLOUncertainties(object):

    def __init__(self, table_filename,
                 lhgrid_filename,
                 member=0,
                 scale_factor=(1.0, 1.0),
                 errortype='auto',
                 pdf_clscale = None):

        self._table_filename = table_filename
        self._lhgrid_filename = lhgrid_filename
        self._member = member
        self._scale_factor = scale_factor

        if errortype is 'auto':
            self._identify_errortype()
        else:
            self._errortype = errortype

        self._pdf_clscale = pdf_clscale

        # Initialize fastNLOReader instance
        self._fnlo = None
        self._init_fastnlo()

        # infos about pdfs and bins
        self._npdfmembers = self._fnlo.GetNPDFMembers()
        self._nobsbins = self._fnlo.GetNObsBin()
        self._ndiffbins = self._fnlo.GetNumDiffBin()

        # Get Differential Bins
        #self._bins_down = numpy.array(self._fnlo.GetLoBin()).transpose()
        #self._bins_up = numpy.array(self._fnlo.GetUpBin()).transpose()

        # Member Cross Sections
        # 1000 member * 1000 obsbins * 10 skalen* 64 / 8 / 100000 = 80 MB in worst case
        # too much: one array per scale
        self._member_crosssections = None
        self._fnlo.SetAlphasMz(0.110)

    def _init_fastnlo(self):
        # fastNLOReader instance
        self._fnlo = fastNLOLHAPDF(self._table_filename,
                                   self._lhgrid_filename, 0)
        self._fnlo.SetLHAPDFMember(self._member)
        # Do this immediately to be able to read out nmember
        self._fnlo.FillPDFCache()

    def reset_cache(self):
        self._member_crosssections = None

    # Public Methods
    # Quickly get all interesting results

    def get_all(self):
        results = {'xsnlo': self.get_central_crosssection(),
                   'scale_uncert': self.get_scale_uncert()}
        if self._errortype in ['MC', 'EV', 'SEV', 'EVVAR']:
            results['pdf_uncert'] = self.get_pdf_uncert()
            results['cov_pdf_uncert'] = self.get_pdf_cov_matrix()
        return results

    ###############
    # Bin Methods #
    ###############

    def get_bins_up(self):
        return self._bins_up

    def get_bins_down(self):
        return self._bins_down

    #########################
    # Cross Section methods #
    #########################

    def get_central_crosssection(self):
        if self._errortype == 'MC':
            return self._get_mean_crosssection()
        elif self._errortype in ['EV', 'SEV', 'EVVAR', 'NONE']:
            return self._get_member_crosssection(member=self._member)

    def get_cross_section(self, member=None, scale_factor=None):
        if scale_factor:
            self._scale_factor = scale_factor
        if self._errortype == 'MC':
            return self._get_mean_crosssection()
        elif self._errortype in ['EV', 'SEV']:
            return self._get_member_crosssection(member=member)

    def get_member_crosssections(self):
        if self._member_crosssections is None:
            self._cache_member_crosssections()
        return self._member_crosssections

    #####################
    # PDF Uncertainties #
    #####################

    def get_pdf_uncert(self, symmetric=False):
        """
        Calls the respective functions to calculate the PDF uncertainties for
        each type of PDF.
        """
        if self._errortype == 'MC':
            return self._get_pdf_std()
        elif self._errortype == 'EV':
            return self._get_pdf_ev(symmetric=symmetric)
        elif self._errortype == 'SEV':
            return self._get_pdf_ev(symmetric=True)
        else:
            return None

    def get_pdf_cov_matrix(self):
        """
        Calls the respective functions to calculate the PDF covariance matrix for
        each type of PDF.
        """
        if self._errortype == 'MC':
            return self._get_pdf_sample_covariance()
        elif self._errortype in ['EV', 'SEV', 'EVVAR']:
            return self._get_pdf_ev_covariance()
        else:
            return None

    #######################
    # Scale Uncertainties #
    #######################

    def get_scale_uncert(self, var='6p', def_murmuf=None):
        """
        Calculate Scale Uncertainties according to assymetric 6 point scale variation
        or symmetric 2 point scale variation.
        @param var: 2-point or 6-point scale variation
        @param def_murmuf: (mur, muf) tuple of default scale factors
        @return: absolute scale uncert
        """
        if def_murmuf is None:
            def_murmuf = self._scale_factor
        if var == '6p':
            scale_variations = [(1.0, 2.0), (1.0, 0.5), (2.0, 1.0),
                                (2.0, 2.0), (0.5, 0.5), (0.5, 1.0)]
        elif var == '2p':
            scale_variations = [(2.0, 2.0), (0.5, 0.5)]
        else:
            scale_variations = []

        def_crosssection = self._get_member_crosssection(scale_factor=def_murmuf)
        scale_uncert = numpy.zeros((2, self._nobsbins))

        for scale_factor in scale_variations:
            scale_crosssection = self._get_member_crosssection(scale_factor=scale_factor)
            scale_uncert[0] = numpy.maximum(scale_uncert[0],
                                            def_crosssection - scale_crosssection)

            scale_uncert[1] = numpy.maximum(scale_uncert[1],
                                            scale_crosssection - def_crosssection)
        return scale_uncert

    #
    #  Private methods
    #

    def _cache_member_crosssections(self):
        """Read Cross Section for all members in PDF """
        self._member_crosssections = numpy.zeros((self._npdfmembers,
                                                  self._nobsbins))

        for member in range(0, self._npdfmembers):
            # self._fnlo.SetLHAPDFMember(member)
            # self._fnlo.CalcCrossSection()
            self._member_crosssections[member] = self._get_member_crosssection(
                member=member)

    def _get_member_crosssection(self, member=None, scale_factor=None):
        """
        Call fastNLOReader to get cross section

        """
        if scale_factor is None:
            scale_factor = self._scale_factor
        if member is None:
            member = self._member
        self._fnlo.SetScaleFactorsMuRMuF(*scale_factor)
        self._fnlo.SetLHAPDFMember(member)
        self._fnlo.CalcCrossSection()
        return numpy.array(self._fnlo.GetCrossSection())

    def _get_mean_crosssection(self):
        """
        Calculate sample mean
        Neglecting first item (often the average)
        """
        if self._member_crosssections is None:
            self._cache_member_crosssections()
        return numpy.mean(self._member_crosssections[1:], axis=0)

    def _get_pdf_std(self):
        """
        Calculate sample standard deviation, but neglecting first member of sample.
        @return:
        """
        if self._member_crosssections is None:
            self._cache_member_crosssections()
        std = numpy.std(self._member_crosssections[1:], axis=0)
        std = numpy.vstack((std, std))
        if self._pdf_clscale is not None:
            std /= self._pdf_clscale
        return std

    def _get_pdf_ev(self, symmetric=False):
        """
        Calculate uncertainties of eigenvector sample. By default asymmetric uncertainties
        are calculated.
        @param symmetric: Calculate symmetric or asymmetric uncertainties.
        @return:
        """
        if self._member_crosssections is None:
            self._cache_member_crosssections()
        pdf_uncert = numpy.zeros((2, self._nobsbins))
        if symmetric is True:
            for i in range(1, self._npdfmembers / 2 + 1):
                pdf_uncert[0] += numpy.square(
                    self._member_crosssections[2 * i - 1] -
                    self._member_crosssections[2 * i])
            pdf_uncert[0] = 0.5 * numpy.sqrt(pdf_uncert[0])
            pdf_uncert[1] = pdf_uncert[0]
        else:
            for i in xrange(1, self._npdfmembers / 2 + 1):
                pdf_uncert[0] += numpy.square(numpy.minimum(numpy.minimum(
                    self._member_crosssections[2 * i - 1] -
                    self._member_crosssections[0],
                    self._member_crosssections[2 * i] -
                    self._member_crosssections[0]), 0.))
                pdf_uncert[1] += numpy.square(numpy.maximum(numpy.maximum(
                    self._member_crosssections[2 * i - 1] -
                    self._member_crosssections[0],
                    self._member_crosssections[2 * i] -
                    self._member_crosssections[0]), 0.))
            pdf_uncert = numpy.sqrt(pdf_uncert)

        if self._pdf_clscale is not None:
            pdf_uncert /= self._pdf_clscale

        return pdf_uncert

    def _get_pdf_sample_covariance(self):
        """
        Calculate sample covariance
        """
        if self._member_crosssections is None:
            self._cache_member_crosssections()

        cov_matrix = numpy.cov(self._member_crosssections[1:], rowvar=0)

        if self._pdf_clscale is not None:
            cov_matrix /= self._pdf_clscale ** 2

        return cov_matrix

    def _get_pdf_ev_covariance(self):
        """
        Calculate covariance of mutually independent but fully correlated
        eigenvectors.
        """

        cov_matrix = numpy.zeros((self._nobsbins, self._nobsbins))
        #TODO: Is it valid for symmetric eigenvectors?
        if self._errortype == 'SEV':
            raise NotImplementedError

        if self._member_crosssections is None:
            self._cache_member_crosssections()
        for i in range(1, (self._npdfmembers / 2) + 1):
            # noinspection PyCallingNonCallable
            cov_matrix += \
                numpy.matrix(self._member_crosssections[2 * i] - self._member_crosssections[2 * i - 1]).getT() * \
                numpy.matrix(self._member_crosssections[2 * i] - self._member_crosssections[2 * i - 1])

        cov_matrix /= 4.

        if self._pdf_clscale is not None:
            cov_matrix /= self._pdf_clscale ** 2

        return cov_matrix

    def _identify_errortype(self):
        """ Identify type of PDF LHgrid file
        MC: Monte carlo ensemble with replicas
        EV: Asymmetric Eigenvectors
        SEV: Symmetric Eigenvectors
        EVVAR: Asymmetric Eigenvectors with additional VAR PDF
        """
        # Scale PDF self._clscale
        if self._lhgrid_filename.startswith('CT10') or self._lhgrid_filename.startswith('cteq'):
            self._errortype = 'EV'
            self._pdf_clscale = 1.645
        elif self._lhgrid_filename.startswith('MSTW'):
            self._errortype = 'EV'
        elif self._lhgrid_filename.startswith('NNPDF'):
            self._errortype = 'MC'
        elif self._lhgrid_filename.startswith('HERAMC'):
            self._errortype = 'MC'
        elif self._lhgrid_filename.startswith('HERA'):
            self._errortype = 'EVVAR'
        elif self._lhgrid_filename.startswith('ABM'):
            self._errortype = 'SEV'
        else:
            raise Exception("Unknown PDF type.")


class fastNLOUncertaintiesAlphas(fastNLOUncertainties):

    def __init__(self, table_filename,
                 lhgrid_filename,
                 member=0,
                 scale_factor=(1.0, 1.0),
                 errortype='auto',
                 pdf_clscale=None,
                 mz=91.1876,
                 alphasmz=0.1184):

        self._mz = mz
        self._alphasmz = 0.1184

        super(fastNLOUncertaintiesAlphas, self).__init__(table_filename,
                                                         lhgrid_filename,
                                                         member=member,
                                                         scale_factor=scale_factor,
                                                         errortype=errortype,
                                                         pdf_clscale=pdf_clscale)

    def _init_fastnlo(self):
        # fastNLOReader instance
        self._fnlo = fastNLOAlphas(self._table_filename,
                                   self._lhgrid_filename, self._member)
        self._fnlo.SetMz(self._mz)
        self._fnlo.SetAlphasMz(self._alphasmz)
        # Do this immediately to be able to read out nmember
        self._fnlo.FillPDFCache()

    def set_mz(self, mz):
        self._mz = mz
        self._fnlo.SetMz(mz)
        self.reset_cache()

    def set_alphasmz(self, alphasmz):
        self._alphasmz = alphasmz
        self._fnlo.SetAlphasMz(alphasmz)
        self.reset_cache()
