import pandas as pd
import os
import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd
import q2
from PyAstronomy import pyasl
from scipy.interpolate import interp1d
import shutil
import emcee
import corner
#import smplotlib
from scipy.optimize import minimize


'''
This code is used to create a MOOG's driver file and run MOOGSILENT.
It is optimized for single-line analysis.
'''

#path
moogpath = os.path.expanduser("~") + '/q2-tools/MOOG-for-q2/./MOOGSILENT'
path     = os.getcwd()


#configuring plot
plotpar = {'axes.labelsize': 25,
           'xtick.labelsize': 20,
           'ytick.labelsize': 20,
           'font.family': 'serif',
           'axes.linewidth': 1,
           'text.usetex': False}
plt.rcParams.update(plotpar)


class quel(object):
    
    def __init__(self):
        self.starid = starid
        
    @classmethod
    def create_model(self, starid, teff, logg, feh, vt, model=None):
        '''
        Creates atmosphere models using q2
        '''
        if not model:
            model = 'odfnew'
        else:
            model = model   #can be marcs
        
        Star = q2.Star(starid, teff=teff, logg=logg, feh=feh, vt=vt)
        Star.get_model_atmosphere(model)
        q2.moog.create_model_in(Star)
        os.system('mv model.in %s.mod' % starid)
    
    
    @classmethod
    def open_spec(self, starid, xmin=None, xmax=None, plot=None):
        '''
        Read spectra in fits format
        '''
        #reading spectra
        wave, flux = pyasl.read1dFitsSpec(starid)
        
        #setting conditions
        if not xmin:
            xmin = wave[0]
        else:
            xmin = xmin
        if not xmax:
            xmax = wave[-1]
        else:
            xmax = xmax
        if not plot:
            plot = 'no'
        else:
            plot = plot
        
        #cutting spectra
        mask = (wave >= xmin) & (wave <= xmax)
        
        #saving spectra
        data = {'wave': wave[mask], 'flux': flux[mask]}
        df = pd.DataFrame(data)
        df.to_csv(starid+'.txt', header=None, index=None, sep='\t', mode='a')
        
        if plot == 'yes':
            #plotting spectra
            plt.figure(figsize=(7, 5))
            plt.plot(wave[mask], flux[mask])
        
            plt.ylabel("Flux")
            plt.xlabel("Wavelength")
            plt.xlim(xmin, xmax)
            plt.tight_layout()
            
        return wave[mask], flux[mask]
    

    @classmethod
    def create_batch_Th(self, starid, vsini, macro_v, wl_start, wl_end, line, resol, abds=None, model_in=None, lines_in=None, observed_in=None, syn_start=None, syn_end=None, step=None, opac=None, atm=None, mol=None, tru=None, lin=None, flu=None, dam=None, v_shift=None, y_shift_add=None, y_shift_mult=None, wl_shift=None, lorentz=None, dark=None, output=None):
        """
        Writes the MOOG driver file batch.par for Th
        """
        if not model_in:
            model_in = starid+'.mod'
        else:
            model_in = model_in
        if not lines_in:
            lines_in = 'thsun90.moog'
        else:
            lines_in = lines_in
        if not observed_in:
            observed_in = starid+'.txt'
        else:
            observed_in = observed_in
        if not abds:
            abds = 'abds_Th.csv'
        else:
            abds = abds
        if not syn_start:
            syn_start = wl_start #- 2
        else:
            syn_start = syn_start
        if not syn_end:
            syn_end = wl_end #+ 2
        else:
            syn_end = syn_end
        if not step:
            step = 0.01
        else:
            step = step
        if not opac:
            opac = 2.0
        else:
            opac = opac
        # MOOG synth options
        if not atm:
            atm = 1
        else:
            atm = atm
        if not mol:
            mol = 1
        else:
            mol = mol
        if not tru:
            tru = 1
        else:
            tru = tru
        if not lin:
            lin = 1
        else:
            lin = lin
        if not flu:
            flu = 0
        else:
            flu = flu
        if not dam:
            dam = 0
        else:
            dam = dam
        #shifts
        if not v_shift:
            v_shift = 0.0
        else:
            v_shift = v_shift
        if not  y_shift_add:
            y_shift_add = 0.0
        else:
            y_shift_add = y_shift_add
        if not y_shift_mult:
            y_shift_mult = 0.0
        else:
            y_shift_mult = y_shift_mult
        if not wl_shift:
            wl_shift  = 0.0
        else:
            wl_shift = wl_shift
        if not lorentz:
            lorentz = 0.0
        else:
            lorentz = lorentz
        if not dark:
            dark = 0.6
        else:
            dark = dark
        if not output:
            output = 'regular'
        else:
            output = output

        #table abundances to perform synthesis
        data = pd.read_csv(path+'/input_data/'+abds)

        #instrumental broadening
        broad_inst = line / resol

        #Output files
        standard_out = 'standard.out'
        summary_out = 'summary.out'
        smoothed_out = 'smoothed.out'

        # Creating batch.par file
        if output == 'regular':
            batchpar = 'batch.par'
        if output == 'upper':
            batchpar = 'batch_upper.par'
        if output == 'lower':
            batchpar = 'batch_lower.par'
        with open(batchpar, 'w') as f:
            f.truncate()
            f.write('synth\n')
            if output == 'regular':
                f.write('standard_out  %s\n' % standard_out)
                f.write('summary_out  %s\n' % summary_out)
                f.write('smoothed_out  %s\n' % smoothed_out)
            if output == 'upper':
                f.write('standard_out  standard_up.out\n')
                f.write('summary_out  summary_up.out\n')
                f.write('smoothed_out  smoothed_up.out\n')
            if output == 'lower':
                f.write('standard_out  standard_lo.out\n')
                f.write('summary_out  summary_lo.out\n')
                f.write('smoothed_out  smoothed_lo.out\n')
            f.write('model_in  %s\n' % model_in)
            f.write('lines_in  %s\n' % lines_in)
            f.write('observed_in  %s\n' % observed_in)
            #f.write('iraf  %s\n' % 1)
            #f.write('iraf_out  %s\n' % 'ABDs.txt')
            f.write('atmosphere  %i\n' % atm)
            f.write('molecules  %i\n' % mol)
            f.write('trudamp  %i\n' % tru)
            f.write('lines    %i\n' % lin)
            f.write('flux/int  %i\n' % flu)
            f.write('damping    %i\n' % dam)
            f.write('freeform  1\n')
            f.write('plot    3\n')
            f.write('abundances %i %i\n' % (len(data['Z']), len(data.axes[1])-4))
            #f.write('abundances  1\n')
            for k in range(len(data['Z'])):
                #if len(data.axes[1])-2 == 1:
                #    f.write('   %i %f\n' % (data['Z'][k], data['ab1'][k]))
                #if len(data.axes[1])-2 == 2:
                #    f.write('   %i %f %f\n' % (data['Z'][k], data['ab1'][k], data['ab2'][k]))
                #if len(data.axes[1])-2 == 3:
                #    f.write('   %i %.3f %.3f %.3f\n' % (data['Z'][k], data['ab1'][k], data['ab2'][k], data['ab3'][k]))
                if output == 'regular':
                    #if len(data.axes[1])-2 == 3:
                    f.write('   %i %f\n' % (data['Z'][k], data['ab'][k]))
                if output == 'upper':
                    f.write('   %i %f\n' % (data['Z'][k], data['upper'][k]))
                if output == 'lower':
                    f.write('   %i %f\n' % (data['Z'][k], data['lower'][k]))
            f.write('isotopes   0  1\n')
            f.write('synlimits\n')
            f.write(' %.1f %.1f %.2f %.1f\n' % (syn_start, syn_end, step, opac))
            f.write('obspectrum  5\n')
            f.write('plotpars  1\n')
            f.write(' %.2f %.2f 0.5 1.05\n' % (wl_start, wl_end))
            f.write('%.4f  %.4f  %.3f  %.3f\n' % (v_shift, wl_shift,
                                                   y_shift_add,
                                                   y_shift_mult))
            f.write(' r  %.2f  %.2f  %.2f  %.2f  %.2f' % (broad_inst, vsini, dark, macro_v, lorentz))
        

    @classmethod
    def Th(self, starid, teff, logg, feh, vt, wl_start, wl_end, line, resol, vsini, macro_v, abds=None, model=None, model_in=None, lines_in=None, observed_in=None, xmin=None, xmax=None, plot=None, syn_start=None, syn_end=None, step=None, opac=None, atm=None, mol=None, tru=None, lin=None, flu=None, dam=None, v_shift=None, y_shift_add=None, y_shift_mult=None, wl_shift=None, lorentz=None, dark=None, plot_li=None, output=None, delete=None):
        '''
        Determine Thorium abundance
        '''
        if not lines_in:
            lines_in = 'thsun90.moog'
        else:
            lines_in = lines_in
        if not plot_li:
            plot_li = 'yes'
        else:
            plot_li = 'no'
        if not output:
            output = 'regular'
        else:
            output = output
        if not delete:
            delete = 'yes'
        else:
            delete = 'no'
        #step0 - moving data from input_data file
        os.system('cp input_data/%s %s' % (lines_in, path))
        os.system('cp input_data/%s %s' % (starid, path))

        #Step1 - read spectra
        xmin = wl_start
        xmax = wl_end
        wave_o, flux_o = self.open_spec(starid, xmin, xmax, plot)

        #Step2 - generate model
        self.create_model(starid, teff, logg, feh, vt, model)

        #Step3 - create moog file
        self.create_batch_Th(starid, vsini, macro_v, wl_start, wl_end, line, resol, abds, model_in, lines_in, observed_in, syn_start, syn_end, step, opac, atm, mol, tru, lin, flu, dam, v_shift, y_shift_add, y_shift_mult, wl_shift, lorentz, dark, output)

        #Step4 - run moog
        #os.system('%s > moog.log 2>&1' % moogpath)
        if output == 'regular':
            os.system('echo batch.par | %s > moog.log 2>&1' % moogpath)
            #reading covolved data
            synth_wlc   = np.loadtxt('smoothed.out', skiprows=2, usecols=(0,))
            synth_fluxc = np.loadtxt('smoothed.out', skiprows=2, usecols=(1,))
        if output == 'upper':
            os.system('echo batch_upper.par | %s > moog.log 2>&1' % moogpath)
            #reading covolved data
            synth_wlc_up   = np.loadtxt('smoothed_up.out', skiprows=2, usecols=(0,))
            synth_fluxc_up = np.loadtxt('smoothed_up.out', skiprows=2, usecols=(1,))
        if output == 'lower':
            os.system('echo batch_lower.par | %s > moog.log 2>&1' % moogpath)
            #reading covolved data
            synth_wlc_lo   = np.loadtxt('smoothed_lo.out', skiprows=2, usecols=(0,))
            synth_fluxc_lo = np.loadtxt('smoothed_lo.out', skiprows=2, usecols=(1,))
        
        if plot_li == 'yes':
            #Plotting/saving a plot
            fig = plt.figure(figsize=(9, 7))
            gs  = gridspec.GridSpec(2, 1, height_ratios=[5, 1.], hspace=0.04)
            gs.update(left=0.12, right=0.955, top=0.98, bottom=0.105)
            ax = plt.subplot(gs[0])
            #plt.plot(synth_wl, synth_flux)
            ax.plot(synth_wlc, synth_fluxc, color='r', label='synthesis')
            ax.scatter(wave_o + wl_shift, flux_o + y_shift_mult, s=60, facecolors='none', edgecolors='b', zorder=2, label='observed spectrum')
            
            #CN
            Th = '#58D68D'
            #plt.text(6707.2052+0.005, np.max(flux_o)+0.025, 'CN', size=15)
            ax.axvspan(4019.11, 4019.15, ymin=0.0, ymax=1.0, alpha=0.4, color=Th)

            ax.get_xaxis().set_visible(False)
            plt.ylim(np.min(flux_o)-0.05, np.max(flux_o)+0.05)
            plt.xlim(synth_wlc[0]+0.502, synth_wlc[-1]-0.502)
            plt.ylabel("Flux")
            #plt.legend(loc="best")

            ax1  = plt.subplot(gs[1])
            #interpolating observed spectra to make difference (diff)
            f    = interp1d(wave_o, flux_o, kind='cubic', fill_value='extrapolate')
            #xnew = np.linspace(wave_o[0], wave_o[-1], num=len(synth_fluxc), endpoint=True)
            ynew = f(synth_wlc)
            diff = synth_fluxc - ynew
            ax1.plot(synth_wlc, diff, color='r', alpha=0.7, linewidth = 2., zorder=2.3)
            ax1.plot((synth_wlc[0], synth_wlc[-1]), (0, 0), color='black', linestyle='--', linewidth = 1.8)
                
            plt.xlim(synth_wlc[0]+0.502, synth_wlc[-1]-0.502)
            plt.ylim(-0.1, 0.1)
            plt.ylabel("S-O", fontsize=20)
            plt.xlabel("Wavelength")
            #plt.tight_layout()
            plt.savefig(starid+'_Th.pdf')

            #interpolation
            #interp_func = interp1d(model_wave, model_flux, kind='cubic', fill_value='extrapolate')
            #interpolated_flux = interp_func(wave_o)

        #savind data
        # Read in the data and skip rows before "MODEL"
        if output == 'regular':
            summary_out = 'summary.out'
            # Read the file
            with open(summary_out, 'r') as f:
                data = f.readlines()
        if output == 'upper':
            summary_out = 'summary_up.out'
            # Read the file
            with open(summary_out, 'r') as f:
                data = f.readlines()
        if output == 'lower':
            summary_out = 'summary_lo.out'
            # Read the file
            with open(summary_out, 'r') as f:
                data = f.readlines()

        # Create a dictionary to store the data
        el  = []
        val = []

        # Iterate over the lines in the file
        for line in data:
            # Check if the line contains an element and its abundance
            if 'element' in line and 'abundance' in line:
                # Extract the element name and abundance value
                element = line.split()[1].strip(':')
                abundance, value = line.strip().split(':')[1].strip().split('=')
                el.append(element)
                val.append(value)
            #initialize data
            dat = {'element': el, 'abundance': val}
            #create Dataframe
            df = pd.DataFrame(dat)
            if output == 'regular':
                df.to_csv('Final_ABDs.csv', index=False)
            if output == 'upper':
                df.to_csv('Final_ABDs_upper.csv', index=False)
            if output == 'lower':
                df.to_csv('Final_ABDs_lower.csv', index=False)
        if delete == 'yes':
            directory = 'results'
            if os.path.exists(directory):
                shutil.rmtree(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        #moving final data to results
        os.system('cp *.pdf *.txt *.csv smoothed*.out results 2>/dev/null')
        
        if output == 'upper':
            os.system('mv smoothed.out smoothed_upper.out')
            os.system('cp smoothed_up.out results')
        if output == 'lower':
            os.system('mv smoothed.out smoothed_lower.out')
            os.system('cp smoothed_lo.out results')
        else:
            os.system('cp smoothed.out results')
            
        #copy changes in abundances
        os.system('cp input_data/abds_Th.csv results/')
                
        #removing unnecessary data
        os.system('rm moog.log *.mod *.txt *.pdf *.par *.out *.fits *csv 2>/dev/null')
        os.system('rm %s' %lines_in)
        
        if output == 'regular':
            return float(val[-1]), synth_wlc, synth_fluxc
        
        if output == 'upper':
            return float(val[-1]), synth_wlc_up, synth_fluxc_up
        
        if output == 'lower':
            return float(val[-1]), synth_wlc_lo, synth_fluxc_lo

    @classmethod
    def Th_err(self, starid, teff, logg, feh, vt, wl_start, wl_end, line, resol, vsini, macro_v, abds=None, model=None, model_in=None, lines_in=None, observed_in=None, xmin=None, xmax=None, plot=None, syn_start=None, syn_end=None, step=None, opac=None, atm=None, mol=None, tru=None, lin=None, flu=None, dam=None, v_shift=None, y_shift_add=None, y_shift_mult=None, wl_shift=None, lorentz=None, dark=None):
        
        if not lines_in:
            lines_in = 'thsun90.moog'
        else:
            lines_in = lines_in
            
        #Step 1
        #Calculating Li abundance
        Th, sywl, syflux = self.Th(starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, lines_in=lines_in, y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', delete='no')
        
        #Step 2
        #Calculating upper limit
        Th_up, sywl_up, syflux_up = self.Th(starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, lines_in=lines_in,  y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', output='upper', delete='no')
        
        #Step 3
        #Calculating lower limit
        Th_lo, sywl_lo, syflux_lo = self.Th(starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, lines_in=lines_in,  y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', output='lower', delete='no')
        
        #Step 4
        #Plotting results
        xmin = wl_start
        xmax = wl_end
        wave_o, flux_o = self.open_spec(path+'/input_data/'+starid, xmin, xmax, plot)
        
        synth_wlc      = np.loadtxt(path+'/results/smoothed.out', skiprows=2, usecols=(0,))
        synth_fluxc    = np.loadtxt(path+'/results/smoothed.out', skiprows=2, usecols=(1,))
        synth_wlc_up   = np.loadtxt(path+'/results/smoothed_up.out', skiprows=2, usecols=(0,))
        synth_fluxc_up = np.loadtxt(path+'/results/smoothed_up.out', skiprows=2, usecols=(1,))
        synth_wlc_lo   = np.loadtxt(path+'/results/smoothed_lo.out', skiprows=2, usecols=(0,))
        synth_fluxc_lo = np.loadtxt(path+'/results/smoothed_lo.out', skiprows=2, usecols=(1,))
        
        fig = plt.figure(figsize=(9, 7))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[5, 1.], hspace=0.04)
        gs.update(left=0.12, right=0.955, top=0.98, bottom=0.105)
        ax = plt.subplot(gs[0])
        #plt.plot(synth_wl, synth_flux)
        ax.plot(synth_wlc, synth_fluxc, color='r', label='synthesis')
        #ax.plot(synth_wlc_up, synth_fluxc_up, color='#76448A', ls='--', lw=2.1, label='upper')
        #ax.plot(synth_wlc_lo, synth_fluxc_lo, color='#76448A', ls='--', lw=2.1, label='lower')
        ax.fill_between(synth_wlc, synth_fluxc_lo, synth_fluxc_up, alpha=0.3)
        ax.scatter(wave_o + wl_shift, flux_o + y_shift_mult, s=60, facecolors='none', edgecolors='b', zorder=2, label='observed spectrum')

        ax.get_xaxis().set_visible(False)
        plt.ylim(np.min(flux_o)-0.05, np.max(flux_o)+0.05)
        plt.xlim(synth_wlc[0]+0.502, synth_wlc[-1]-0.502)
        plt.ylabel("Flux")
        #plt.legend(loc="best")
        
        ax1  = plt.subplot(gs[1])
        #interpolating observed spectra to make difference (diff)
        f    = interp1d(wave_o, flux_o)
        xnew = np.linspace(wave_o[0], wave_o[-1], num=len(synth_fluxc), endpoint=True)
        ynew = f(xnew)
        diff = synth_fluxc - ynew
        ax1.plot(synth_wlc, diff, color='r', alpha=0.7, linewidth = 2., zorder=2.3)
        ax1.plot((synth_wlc[0], synth_wlc[-1]), (0, 0), color='black', linestyle='--', linewidth = 1.8)
                
        plt.xlim(synth_wlc[0]+0.502, synth_wlc[-1]-0.502)
        plt.ylim(-0.4, 0.4)
        plt.ylabel("S-O", fontsize=20)
        plt.xlabel("Wavelength")
        #plt.tight_layout()
        plt.savefig(starid+'_Th_err.pdf')
        
        #Step 5
        #Saving data
        os.system('mv *.pdf results 2>/dev/null')
        data    = pd.read_csv(path+'/results/Final_ABDs.csv')
        data_up = pd.read_csv(path+'/results/Final_ABDs_upper.csv')
        data_lo = pd.read_csv(path+'/results/Final_ABDs_lower.csv')
        
        #data['ab_upper']    = data_up['abundance']
        #data['ab_lower']    = data_lo['abundance']
        #data['delta_upper'] = data['ab_upper'] - data['abundance']
        #data['delta_lower'] = data['abundance'] - data['ab_lower']
        delta_upper = Th - Th_up
        delta_lower = Th_lo - Th
        
        data.to_csv(path+'/results/Th_abundance.csv', index=False)
        os.system('rm results/Final*.csv 2>/dev/null')
        
        return Th, delta_upper, delta_lower

    ###### BAYESIAN METHOD ######
    @classmethod
    def Th_mcmc(self, starid, teff, logg, feh, vt, wl_start, wl_end, line, resol, vsini, macro_v, SNR, ab_lm=None, nsteps=None, abds=None, model=None, model_in=None, lines_in=None, observed_in=None, xmin=None, xmax=None, plot=None, syn_start=None, syn_end=None, step=None, opac=None, atm=None, mol=None, tru=None, lin=None, flu=None, dam=None, v_shift=None, y_shift_add=None, y_shift_mult=None, wl_shift=None, lorentz=None, dark=None, plot_li=None, output=None, delete=None):
        
        '''
        Determine Thorium abundance using Bayesian Method
        '''
        lines_in = lines_in or 'thsun90.moog'
        plot_li = plot_li or 'yes'
        output = output or 'regular'
        delete = delete or 'yes'
        y_shift_mult = y_shift_mult or 0.0
        ab_lm = ab_lm if ab_lm is not None else 0.3
        nsteps = nsteps if nsteps is not None else 1000
        
        #step0 - moving data from input_data file
        os.system('cp input_data/%s %s' % (lines_in, path))
        os.system('cp input_data/%s %s' % (starid, path))

        #Step1 - read spectra
        xmin = wl_start
        xmax = wl_end
        wave_o, flux_o = self.open_spec(starid, xmin, xmax, plot)
        flux_o_err = flux_o/SNR
        
        # Define the model function
        def model(params, starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, abds='abds_Th.csv', y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', delete='no'):
            
            Thorium = params
            df = pd.read_csv(path+'/input_data/abds_Th.csv')
            # Change the 'ab' column for a given element
            df.loc[df['el'] == 'Th', 'ab'] = Thorium
            #df.loc[df['el'] == 'Co', 'ab'] = Cobalt
            df.to_csv(path+'/input_data/abds_Th_new.csv', index=False)
            
            fakeab, model_wave, model_flux = self.Th(starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, abds='abds_Th_new.csv', y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', delete='no')
            
            #interpolation
            interp_func = interp1d(model_wave, model_flux, kind='cubic', fill_value='extrapolate')
            interpolated_flux = interp_func(wave_o)
            '''
            fig = plt.figure(figsize=(9, 7))
            gs  = gridspec.GridSpec(2, 1, height_ratios=[5, 1.], hspace=0.04)
            gs.update(left=0.12, right=0.955, top=0.98, bottom=0.105)
            ax = plt.subplot(gs[0])
        
            ax.plot(wave_o, interpolated_flux, color='r', label='synthesis')
            ax.scatter(wave_o, flux_o-0.015, s=80, facecolors='none', edgecolors='b', zorder=2, label='observed spectrum')
            '''
            return interpolated_flux
        
        def chi_square(params):
            #Thorium = params
            model_flux = model(params, starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, abds='abds_Th.csv', y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', delete='no')
            '''
            fig = plt.figure(figsize=(9, 7))
            gs  = gridspec.GridSpec(2, 1, height_ratios=[5, 1.], hspace=0.04)
            gs.update(left=0.12, right=0.955, top=0.98, bottom=0.105)
            ax = plt.subplot(gs[0])
        
            ax.plot(wave_o, model_flux, color='r', label='synthesis')
            ax.scatter(wave_o, flux_o-0.015, s=80, facecolors='none', edgecolors='b', zorder=2, label='observed spectrum')
            ax.axvspan(4019.12, 4019.13, ymin=0.0, ymax=1.0, alpha=0.4, color='#58D68D')
            plt.xlim(4018.9, 4019.4)
            plt.ylim(0.25, 1.05)
            '''
            #Focusing only in thorium
            mask = (wave_o >= 4019.11) & (wave_o <= 4019.15)   #following Botelho
            #print (flux_o[mask]-0.015, model_flux[mask], flux_o[mask] -0.015 - model_flux[mask])
            #print (wave_o[mask])
            chi2 = np.sum(((flux_o[mask] + y_shift_mult - model_flux[mask]) / flux_o_err[mask]) ** 2)
            return chi2

        Th_range = np.linspace(-ab_lm, ab_lm, 50)
        chi_square_values = [chi_square(i) for i in Th_range]
        best_fit_abundance = Th_range[np.argmin(chi_square_values)]

        #best solution
        params = best_fit_abundance
        model_flux = model(params, starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, abds='abds_Th.csv', y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', delete='no')
        '''
        fig = plt.figure(figsize=(9, 7))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[5, 1.], hspace=0.04)
        gs.update(left=0.12, right=0.955, top=0.98, bottom=0.105)
        ax = plt.subplot(gs[0])
        
        ax.plot(wave_o, model_flux, color='r', label='synthesis')
        ax.scatter(wave_o, flux_o + y_shift_mult, s=80, facecolors='none', edgecolors='b', zorder=2, label='observed spectrum')
        for i in Th_range:
            params = i
            model_flux = model(params, starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, abds='abds_Th.csv', y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', delete='no')
            mask = (wave_o >= 4019.05) & (wave_o <= 4019.2)
            ax.plot(wave_o[mask], model_flux[mask], 'k--', linewidth=0.6, zorder=0)
            #mask = (wave_o >= 4019.11) & (wave_o <= 4019.15)
            #ax.scatter(wave_o[mask], model_flux[mask], s=40, zorder=0)
        
        plt.xlim(4018.9, 4019.4)
        plt.ylim(0.25, 1.05)
        '''
        #Plot the chi-square values
        fig = plt.figure(figsize=(10, 6))
        plt.plot(Th_range, chi_square_values, marker='o', linestyle='-', color='b')
        plt.axvline(x=best_fit_abundance, color='r', linestyle='--', label=f'Best-fit abundance: {best_fit_abundance:.2f}')
        plt.xlabel('Abundance')
        plt.ylabel('Chi-square')
        plt.title('Chi-square vs. Abundance')
        
        # MCMC to estimate the uncertainties
        def log_likelihood(params):
            return -0.5 * chi_square(params)

        def log_prior(params):
            if -ab_lm < params < ab_lm:
                return 0.0
            return -np.inf

        def log_probability(params):
            lp = log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(params)

        initial = best_fit_abundance
        ndim, nwalkers = 1, 5
        pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        sampler.run_mcmc(pos, nsteps, progress=True)

        samples = sampler.get_chain(discard=100, thin=15, flat=True)
        best_fit_abundance = np.median(samples)
        uncertainty = np.std(samples)
        
        # Plot the corner plot
        corner.corner(
            samples,
            labels=[
                'Thorium'
            ],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            )
        
        #getting sigma
        #mcmc_th = np.percentile(samples[:, 0], [16, 50, 84])
        #q_th = np.diff(mcmc_th)
        
        #best solution
        params = best_fit_abundance
        model_fluxi = model(params, starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, abds='abds_Th.csv', y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', delete='no')
        
        fig = plt.figure(figsize=(9, 7))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[5, 1.], hspace=0.04)
        gs.update(left=0.12, right=0.955, top=0.98, bottom=0.105)
        ax = plt.subplot(gs[0])
        
        ax.plot(wave_o, model_fluxi, color='r', label='synthesis')
        ax.scatter(wave_o, flux_o + y_shift_mult, s=60, facecolors='none', edgecolors='b', zorder=2, label='observed spectrum')
        ax.axvspan(4019.11, 4019.15, ymin=0.0, ymax=1.0, alpha=0.4, color='#58D68D')
        
        plt.xlim(4018.9, 4019.4)
        plt.ylim(0.25, 1.05)
        plt.ylabel("Normalized Flux", fontsize=20)
        plt.xlabel("Wavelength")
        #plt.tight_layout()
        plt.savefig('best_solution.pdf')
        
        return best_fit_abundance
        
    
    ###### chi2 minimization ######
    @classmethod
    def Th_ximin(self, starid, teff, logg, feh, vt, wl_start, wl_end, line, resol, vsini, macro_v, SNR, ab_lm=None, nsteps=None, abds=None, model=None, model_in=None, lines_in=None, observed_in=None, xmin=None, xmax=None, plot=None, syn_start=None, syn_end=None, step=None, opac=None, atm=None, mol=None, tru=None, lin=None, flu=None, dam=None, v_shift=None, y_shift_add=None, y_shift_mult=None, wl_shift=None, lorentz=None, dark=None, plot_li=None, output=None, delete=None):
        
        '''
        Determine Thorium abundance using chi2
        '''
        lines_in = lines_in or 'thsun90.moog'
        plot_li = plot_li or 'yes'
        output = output or 'regular'
        delete = delete or 'yes'
        y_shift_mult = y_shift_mult or 0.0
        ab_lm = ab_lm if ab_lm is not None else 0.3
        nsteps = nsteps if nsteps is not None else 1000
        
        #step0 - moving data from input_data file
        os.system('cp input_data/%s %s' % (lines_in, path))
        os.system('cp input_data/%s %s' % (starid, path))

        #Step1 - read spectra
        xmin = wl_start
        xmax = wl_end
        wave_o, flux_o = self.open_spec(starid, xmin, xmax, plot)
        flux_o_err = flux_o/SNR
        
        # Define the model function
        def model(params, starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, abds='abds_Th.csv', y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', delete='no'):
            
            Thorium = params
            df = pd.read_csv(path+'/input_data/abds_Th.csv')
            # Change the 'ab' column for a given element
            df.loc[df['el'] == 'Th', 'ab'] = Thorium
            #df.loc[df['el'] == 'Co', 'ab'] = Cobalt
            df.to_csv(path+'/input_data/abds_Th_new.csv', index=False)
            
            fakeab, model_wave, model_flux = self.Th(starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, abds='abds_Th_new.csv', y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', delete='no')
            
            #interpolation
            interp_func = interp1d(model_wave, model_flux, kind='cubic', fill_value='extrapolate')
            interpolated_flux = interp_func(wave_o)
  
            return interpolated_flux
        
        def chi_square(params):
            #Thorium = params
            model_flux = model(params, starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, abds='abds_Th.csv', y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', delete='no')
 
            #Focusing only in thorium
            mask = (wave_o >= 4019.11) & (wave_o <= 4019.15)   #following Botelho
            chi2 = np.sum(((flux_o[mask] + y_shift_mult - model_flux[mask]) / flux_o_err[mask]) ** 2)
            return chi2

        Th_range = np.linspace(-ab_lm, ab_lm, 100)
        chi_square_values = [chi_square(i) for i in Th_range]
        best_fit_abundance = Th_range[np.argmin(chi_square_values)]
                
        #best solution
        params = best_fit_abundance
        model_flux = model(params, starid=starid, teff=teff, logg=logg, feh=feh, vt=vt, wl_start=wl_start, wl_end=wl_end, line=line, resol=resol, vsini=vsini, macro_v=macro_v, abds='abds_Th.csv', y_shift_mult=y_shift_mult, wl_shift=wl_shift, plot_li='no', delete='no')
    
        #Plot the chi-square values
        fig = plt.figure(figsize=(10, 6))
        plt.plot(Th_range, chi_square_values, marker='o', linestyle='-', color='b')
        plt.axvline(x=best_fit_abundance, color='r', linestyle='--', label=f'Best-fit abundance: {best_fit_abundance:.2f}')
        plt.xlabel('Abundance')
        plt.ylabel('Chi-square')
        plt.title('Chi-square vs. Abundance')
        
        fig = plt.figure(figsize=(9, 7))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[5, 1.], hspace=0.04)
        gs.update(left=0.12, right=0.955, top=0.98, bottom=0.105)
        ax = plt.subplot(gs[0])
        
        ax.plot(wave_o, model_flux, color='r', label='synthesis')
        ax.scatter(wave_o, flux_o + y_shift_mult, s=60, facecolors='none', edgecolors='b', zorder=2, label='observed spectrum')
        ax.axvspan(4019.11, 4019.15, ymin=0.0, ymax=1.0, alpha=0.4, color='#58D68D')
        
        plt.xlim(4018.9, 4019.4)
        plt.ylim(0.25, 1.05)
        plt.ylabel("Normalized Flux", fontsize=20)
        plt.xlabel("Wavelength")
        
        
        
        return best_fit_abundance