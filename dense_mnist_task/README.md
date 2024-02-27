example run:

python /home/mila/r/roy.eyono/danns_eg/dense_mnist_task/train.py --opt.lr=0.01 --opt.wd=1e-4 --opt.inhib_momentum=0.9 --opt.momentum=0.5 --opt.inhib_lrs.wei=0.5 --opt.inhib_lrs.wix=1 --train.batch_size=32 --exp.save_results=True --train.dataset=rm_mnist --train.epochs=50

# Homeostatic DANN
python train.py --opt.lr=0.1 --opt.wd=1e-6 --opt.inhib_momentum=0.9 --opt.momentum=0.5 --opt.inhib_lrs.wei=0.5 --opt.inhib_lrs.wix=1e-4 --train.batch_size=32 --exp.name="homeo_dann"

# Vanilla DANN
python train.py --opt.lr=1 --opt.wd=1e-6 --opt.inhib_momentum=0 --opt.momentum=0.5 --opt.inhib_lrs.wei=1e-4 --opt.inhib_lrs.wix=0.5 --train.batch_size=32 --exp.name="non_homeo_dann"



                    # bool_acc = np.array([int(x) for x in out.argmax(1).eq(labs)])
                    # softmax_score = np.array([max(x).cpu().item() for x in out])
                    # # Zero if wrong, softmax confident if correct
                    # lst_color.extend(np.multiply(bool_acc, softmax_score))
    # elif p.train.dataset=='rm_mnist' or p.train.dataset=='rm_fashionmnist' :
    #     with torch.no_grad():
    #         for ims, labs in loaders['ood']:
    #                 with autocast():
    #                     out = model(ims)
    #                     bool_acc = np.array([int(x) for x in out.argmax(1).eq(labs)])
    #                     softmax_score = np.array([max(x).cpu().item() for x in out])
    #                     # Zero if wrong, softmax confident if correct
    #                     lst_color.extend(np.multiply(bool_acc, softmax_score))


plt.figure(figsize=(5, 5))

    # Create a violin plot
    plt.violinplot(var_data, showmeans=False, showmedians=True)
    plt.title('Variance across distributions')
    #plt.xlabel('Distributions')
    plt.ylabel('Variance')
    #plt.yscale('log')

    # Customize x-axis labels if needed
    plt.xticks(np.arange(1, len(var_data) + 1), [f'{var_data_lbl[i]}' for i in range(0, len(var_data))])

    if p.exp.save_results:
        plt.savefig('ood_violin_plot.pdf')