from AttentiveFP.build_dataset import built_data_and_save_for_pka
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="the name of the target dataset", type=str, default="pka_acidic_2750")
    args = parser.parse_args()

    input_csv = '../data/origin_data/' + args.dataset + '.csv'

    output_g_attentivefp_bin = '../data/Attentivefp_graph_data/' + args.dataset + '_graph.bin'
    output_csv = '../data/Attentivefp_graph_data/' + args.dataset + '_group.csv'

    built_data_and_save_for_pka(
        origin_path=input_csv,
        save_g_attentivefp_path=output_g_attentivefp_bin,
        smiles_path=output_csv
    )

