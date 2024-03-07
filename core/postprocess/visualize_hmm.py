import matplotlib.pyplot as plt

FONT_SIZE = 12
FONT_SIZE_SMALL = 12
DPI = 300


class HmmVisualizer:
    @staticmethod
    def save_bar_plot(
            model_filename,
            df_data,
            data_name,
            prob_name,
            legend=True,
            figsize=(9, 6),
            subplots=False,
            layout=None,
    ):
        plt.figure()
        plt.rcParams.update({'font.size': FONT_SIZE_SMALL})
        if legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, ncol=2)
        if layout:
            df_data.plot.bar(figsize=figsize, fontsize=FONT_SIZE_SMALL, subplots=subplots, layout=layout, legend=legend, sharex=False, sharey=True, color='black')
        else:
            df_data.plot.bar(figsize=figsize, fontsize=FONT_SIZE_SMALL, subplots=subplots, legend=legend, sharex=False, sharey=True, color='black')
        out_filename = str(model_filename)
        if bool(prob_name):
            out_filename += '-{}'.format(prob_name)
        if bool(data_name):
            out_filename += '-{}'.format(data_name)
        plt.savefig('{}.pdf'.format(out_filename), bbox_inches='tight', dpi=DPI)
        plt.close('all')
