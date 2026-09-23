BLUE, ORANGE, GRAY = '#256C9C','#D57627','#969CA3'
# The saved width is the manuscript's full text width: font sizes remain
# physical points instead of becoming smaller when LaTeX includes the PDF.
fig,axes=plt.subplots(1,3,figsize=(5.5,2.8),gridspec_kw={'width_ratios':[1,1,1.06]})
fig.subplots_adjust(left=.11,right=.99,top=.71,bottom=.32,wspace=.53)
fig.text(.02,.975,'MJD waveforms and energy',fontsize=10,weight=title_weight,ha='left',va='top')
fig.text(.02,.900,'Clean test events; detector 133, run 50928',fontsize=8.5,ha='left')

line_handles=[]
zoom=axes[1].inset_axes([.47,.13,.50,.32])
for name,color,style in [('Low energy',BLUE,'-'),('High energy',ORANGE,'--')]:
    rows=[row for row in examples if row['cohort']==name]
    assert len(rows)==3800
    t=np.array([int(row['sample_index']) for row in rows])
    v=np.array([float(row['baseline_subtracted_adc']) for row in rows])
    u=np.array([float(row['normalized_amplitude']) for row in rows])
    en=float(rows[0]['energy_keV'])
    for ax,val in zip(axes[:2],(v,u)):
        ax.plot(t,val,color=color,ls=style,lw=.85 if name=='Low energy' else 1.0,zorder=3)
    zoom.plot(t[:200],u[:200],color=color,ls=style,lw=.65)
    line_handles.append(Line2D([0],[0],color=color,ls=style,lw=1.2,label=f'{name}: {en:.0f} keV'))
fig.legend(handles=line_handles,loc='upper left',bbox_to_anchor=(.01,.875),
           ncol=2,frameon=False,fontsize=8,handlelength=2.1,columnspacing=1.7)
for ax,title,ylabel in zip(axes[:2],['(a) Baseline-subtracted','(b) Normalized'],['ADC counts','Normalized amplitude']):
    ax.set_title(title,loc='left',pad=7)
    ax.set_xlabel('Sample index',labelpad=3)
    ax.set_ylabel(ylabel,labelpad=3)
    ax.set_xlim(0,3799);ax.set_xticks([0,2000,3800])
    ax.grid(axis='y',lw=.5,color='#E2E5E8',zorder=0)
axes[0].set_ylim(-110,3010);axes[0].set_yticks([0,1000,2000,3000])
axes[1].set_ylim(-.06,1.08);axes[1].set_yticks([0,.5,1.])
zoom.set_xlim(0,199);zoom.set_ylim(-.025,.025)
zoom.set_xticks([0,199]);zoom.set_yticks([-.02,.02]);zoom.tick_params(labelsize=7,pad=1,length=2)
zoom.axhline(0,color='#B0B4BA',lw=.45,zorder=0)
zoom.set_title('Baseline zoom',fontsize=7.5,pad=3)

ax=axes[2]
for name,color,marker in [('Middle energy',GRAY,'o'),('Low energy',BLUE,'o'),('High energy',ORANGE,'^')]:
    mask=cohort==name
    ax.scatter(energy[mask],rms[mask],s=7,color=color,marker=marker,alpha=.64,linewidths=0,rasterized=True)
ax.set_xscale('log');ax.set_yscale('log')
ax.set_xlabel('Energy (keV)',labelpad=3)
ax.set_ylabel('Relative baseline RMS',labelpad=3)
ax.set_title('(c) Population',loc='left',pad=7)
ax.grid(which='major',color='#E2E5E8',lw=.5)
ax.text(.965,.965,rf'$\rho={rho:.3f}$',transform=ax.transAxes,ha='right',va='top',fontsize=8)
fig.text(.02,.130,'(a,b) Examples nearest cohort energy medians. (c) All 410 events:',fontsize=8)
fig.text(.02,.082,'low / high quartiles: 103 each; middle half: 204 (gray). Normalization',fontsize=8)
fig.text(.02,.034,'precedes model-specific representations; RMS uses samples 0–199.',fontsize=8,color='#53595F')
