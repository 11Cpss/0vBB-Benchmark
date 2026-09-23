BLUE, ORANGE, MID = '#256C9C', '#D57627', '#92989F'
colors = [BLUE, ORANGE]
sample = np.arange(waves.shape[1])
fig, axs = plt.subplots(2,2,figsize=(5.225,4.1),sharex=True,sharey='row')
fig.subplots_adjust(left=.15,right=.985,bottom=.245,top=.755,hspace=.43,wspace=.17)
fig.suptitle('MJD low- and high-energy waveforms',x=.02,y=.98,ha='left',fontsize=10,weight=title_weight)
fig.text(.02,.918,f'Clean test events; detector {detector}, run {run}; one event near each',fontsize=8.5)
fig.text(.02,.882,'cohort median energy.',fontsize=8.5)
fig.text(.02,.846,f'Low: {energy[low].min():.1f}–{q25:.1f} keV (n={low.sum()}); high: {q75:.1f}–{energy[high].max():.1f} keV (n={high.sum()})',fontsize=8.5)
for col,j in enumerate(examples):
    title=f'{cohort[j]}: {energy[j]:.1f} keV'
    for row, values in enumerate([centered[j], normalized[j]]):
        ax=axs[row,col]
        ax.plot(sample,values,color=colors[col],lw=1)
        ax.axhline(0,color='#6E747B',lw=.6,zorder=0)
        ax.set_xlim(0,3799)
        ax.grid(axis='y',color='#E5E8EB',lw=.6)
        ax.set_title(title if row==0 else 'Amplitude-normalized input',loc='left')
    axs[0,col].text(.97,.23,f'Event ID {meta["id"][indices[j]]}',transform=axs[0,col].transAxes,ha='right',fontsize=8)
    # The same original samples, expanded vertically and horizontally.
    inset=axs[1,col].inset_axes([.49,.17,.46,.30])
    inset.plot(sample[:200],normalized[j,:200],color=colors[col],lw=.75)
    inset.axhline(0,color='#9B9B9B',lw=.5)
    inset.set_xlim(0,199)
    limit=max(float(np.abs(normalized[examples,:200]).max())*1.15,.002)
    inset.set_ylim(-limit,limit)
    inset.set_xticks([0,199]); inset.tick_params(labelsize=7,pad=1,length=2)
    inset.set_title('Baseline zoom',fontsize=7.5,pad=3)
axs[0,0].set_ylabel('Baseline-subtracted\namplitude (ADC counts)')
axs[0,0].set_yticks([0,1000,2000])
axs[1,0].set_ylabel('Normalized amplitude')
for ax in axs[-1]: ax.set_xlabel('Sample index')
axs[1,0].set_ylim(-.065,1.08)
fig.text(.02,.115,'Subtract the mean of samples 0–199; divide each event by its',fontsize=8)
fig.text(.02,.077,'maximum absolute amplitude. Insets show samples 0–199.',fontsize=8)
fig.text(.02,.039,'Row axes and insets share limits. Sampling interval is not assumed.',fontsize=8,color='#50555A')
