import argparse
import os
import torch
# define common parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--train_sample', type=int, default=150,
                        help='train sample num.')
    parser.add_argument('--valid_sample', type=int, default=100,
                        help='valid sample num.')
    parser.add_argument('--test_sample', type=int, default=100,
                        help='test sample num.')
    parser.add_argument('--default_train_size', type=int, default=150,
                        help='Default Train size.')
    parser.add_argument('--train_size', type=int, default=150,
                        help='Train size.')
    parser.add_argument('--valid_size', type=int, default=100,
                        help='Valid size.')
    parser.add_argument('--test_size', type=int, default=100,
                        help='Test size.')
    parser.add_argument('--train_ratio', type=float, default=3/7,
                        help='Train ratio.')
    parser.add_argument('--valid_ratio', type=float, default=2/7,
                        help='Valid ratio.')
    parser.add_argument('--test_ratio', type=float, default=2/7,
                        help='Test ratio.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='training epoch')
    parser.add_argument('--early_stopping', type=int, default=1,
                        help='The patience of earlystopping. Do not adopt the earlystopping when it equals 0.')
    parser.add_argument('--attr', type=int, default = 0,
                        help='Use attribute when it equals to 1.')
    parser.add_argument('--patience', type=int, default = 50,
                        help='Patience for early stopping.')
    parser.add_argument('--save_data', type=int, default=0,
                        help='Save processed data when it equals to 1.')
    parser.add_argument('--save_per_epoch', type=int, default=1,
                        help='Save information per epoch.')
    parser.add_argument('--exp_mod', type=int, default=1,
                        help='Choose experiments. 0: transductive; 1: inductive; 2: hybrid; 3: query size; 4: scalability; 5: dynamic; 6: attribute; 7: vary train size; 8: mask groundtruth')
    parser.add_argument('--exp_param', type=int, default=0,
                        help='The parameter for each experiment.')
    parser.add_argument('--params', type=int, nargs='+', default=[-1],
                        help='The parameter list for each experiment.')
    parser.add_argument('--com_limit', type=int, default= 3,
                        help='Filter the ground-truth communities with size smaller than threshold.')
    parser.add_argument('--loc_model', action='store_true',
                        help='Train a model to predict the threshold.')

    parser.add_argument('--train_query_file', type=str, default="1_induct_train_query",
                        help='The file name of training query.')
    parser.add_argument('--train_gt_file', type=str, default="1_induct_train_gt",
                        help='The file name of training ground-truth.')
    parser.add_argument('--valid_query_file', type=str, default="1_induct_valid_query",
                        help='The file name of validation query.')
    parser.add_argument('--valid_gt_file', type=str, default="1_induct_valid_gt",
                        help='The file name of validation ground-truth.')
    parser.add_argument('--test_query_file', type=str, default="1_induct_test_query",
                        help='The file name of testing query.')
    parser.add_argument('--test_gt_file', type=str, default="1_induct_test_gt",
                        help='The file name of testing ground-truth.')

    parser.add_argument('--dataset_pyg', type=str, nargs='+', default=["cora", "citeseer", "photo", "cs", "Physics"],
                        help='All datasets from pyg.')
    parser.add_argument('--dataset_snap', type=str, nargs='+', default=["email","dblp_snap", "amazon", "youtube", "livejournal"],
                        help='All datasets from SNAP.')

    parser.add_argument('--learning_models', type=str, nargs='+', default=['ICS_GNN', 'QD_GNN', 'CommunityAF', 'CommunityDF', 'COCLEP', 'CSFormer', 'TransZero'],
                        help='All models.')

    parser.add_argument('--data_path', type=str, default='../data/datasets/',
                        help='The path for the data')
    parser.add_argument('--root_path', type=str, default='../',
                        help='The path for the root of project')

    parser.add_argument('--model_path', type=str, default='../output/model/',
                        help='The path for the model to save')
    parser.add_argument('--result_path', type=str, default='../output/result/',
                        help='The path for the result of different model')
    parser.add_argument('--train_path', type=str, default='../output/train/',
                        help='The path for the information during training')
    parser.add_argument('--process_path', type=str, default='../output/process/',
                        help='The path for the processed data of each model')

    parser.add_argument('--diam_limit', type=int, default=500,
                        help='The upper bound of vertices to compute exact diamter')
    parser.add_argument('--max_com_size', type=int, default=1000,
                        help='The max size of the community (Hyperparameters), use validation community to choose')
    parser.add_argument('--non_learning_models', type=str, nargs='+',
                        default=[ "QDC", "LM", "SGM", "DMCS", "PPR", "kcore", "ktruss", "kclique", "kecc"],
                        help='All non-learning community search methods.')
    parser.add_argument('--method_name', type=str, default='PPR',
                        help='The running model.')
    parser.add_argument('--phase', type=str, default='all',
                        help='all, train, rec_generate_all, rec_generate_train, rec_query_all, rec_query_train')
    parser.add_argument('--generate_mod', type=str, default='recLabel', choices=['recLabel', 'recValid', 'recExact'],
                        help='search the best model and parameter for train/valid/ test query in recommendation phase')

    parser.add_argument('--k_based_methods', type=str, nargs='+', default=['ktruss', 'kclique'],
                        help='Methods that need k-parameter tuning (default: ktruss kclique)')
    parser.add_argument('--index_based_methods', type=str, nargs='+', default=['kcore', 'ktruss', 'kclique', 'kecc'],
                        help='Methods that need index construction (default: ktruss kclique)')

    parser.add_argument('--experiment_types', type=str, nargs='+', default=['transduct','induct', 'hybrid', 'query_size', 'scalability', 'dynamic', 'attribute', 'train_size', 'mask_gt'],
                        help='Different types of experiment (default: hybrid)')

    parser.add_argument('--single_query', type=str, nargs='+',
                        default=["COCLEP","CGNP","ICS_GNN","CommunityAF","CommunityDF","CSFormer", "LM", "PPR", "ktruss"],
                        help='Methods that only support single query')
    parser.add_argument('--fix', action='store_true',
                        help='input parameter to control the update in the dynamic experiment')
    parser.add_argument('--all_models', type=str, nargs='+',
                        default=["QDC", "LM", "SGM", "DMCS", "PPR", "kcore", "ktruss", "kclique", "kecc", "ICS_GNN", "QD_GNN", "CommunityAF", "CommunityDF", "COCLEP", "TransZero", "CSFormer"],
                        help='All community search methods.')

    parser.add_argument('--recommend_model', type=str, default='TransZero',
                        help='The model to generate recommend label.')

    parser.add_argument('--rec_topk', type=int, default=3,
                        help='Recommend top-k models.')
    parser.add_argument('--rec_task', type=str, default='recommend',
                        help='Recommend task: recommend or generate.')
    parser.add_argument('--gnn_topn', type=int, default=500,
                        help='The topn of candidate subgraph for recommendation.')
    parser.add_argument('--rec_exp_mod', type=int, default=1,
                        help='Control the experiment mode for recommendation.')
    parser.add_argument('--ablation_mode', type=str, default=None,
                        help='Ablation study mode. Options: "gnn"')
    parser.add_argument('--lambda_param_KL', type=float, default=1.0,
                        help='lambda_param_KL for parameter recommendation.')
    parser.add_argument('--lambda_model_KL', type=float, default=0.1,
                        help='lambda_model_KL for model recommendation.')

    parser.add_argument('--eval_bestQ', action='store_true',
                        help='evaluate the percentage of best query for the community search methods.')
    parser.add_argument('--eval_gt_property', action='store_true',
                        help='evaluate the impact of ground-truth communities for the community search methods.')
    parser.add_argument('--eval_recommend', action='store_true',
                        help='evaluate the result of recommendation.')
         

    args = parser.parse_args()

    root_path = os.path.dirname(os.path.abspath(__file__))
    args.root_path = root_path
    # print(f"root_path: {root_path}")

    args.data_path = os.path.abspath(os.path.join(root_path, "data/datasets/"))
    # print(f"args.data_path: {args.data_path}")

    args.model_path = os.path.join(root_path, "output/model/")
    os.makedirs(args.model_path, exist_ok=True)
    # print(f"args.model_path: {args.model_path}")

    args.result_path = os.path.join(root_path, "output/result/")
    os.makedirs(args.result_path, exist_ok=True)
    # print(f"args.result_path: {args.result_path}")

    args.train_path = os.path.join(root_path, "output/train/")
    os.makedirs(args.train_path, exist_ok=True)
    # print(f"args.train_path: {args.train_path}")

    args.process_path = os.path.join(root_path, "output/process/")
    os.makedirs(args.process_path, exist_ok=True)
    # print(f"args.process_path: {args.process_path}")

    return args

if __name__ == "__main__":
    args = get_args()