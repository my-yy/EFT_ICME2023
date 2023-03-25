import os
from utils import myparser, seed_util, wb_util, model_util, pickle_util \
    , eval_util, model_selector, unsup_nce, cuda_util, worker_util
from models import transformer
import torch
from loaders import loader
from torch.utils.data import DataLoader


def do_step(epoch, step, tup):
    optimizer.zero_grad()
    if len(tup) == 2:
        data, data_info = tup
    else:
        data = tup
        data_info = {}

    data = cuda_util.to_cuda(data)
    v_emb, f_emb = model(data)
    loss1, logits1 = loss_func(v_emb, f_emb)
    loss2, logits2 = loss_func(f_emb, v_emb)

    loss_metric = loss1 + loss2

    loss = loss_metric

    loss.backward()
    optimizer.step()
    return loss.item(), {**data_info}


def train():
    step = 0
    model.train()

    for epo in range(args.epoch):
        wb_util.log({"train/epoch": epo})
        for data in train_iter:
            loss, info = do_step(epo, step, data)
            step += 1
            if step % 50 == 0:
                obj = {
                    "train/step": step,
                    "train/loss": loss,
                }
                obj = {**obj, **info}
                print(obj)
                wb_util.log(obj)

            if step > 0 and step % args.eval_step == 0:
                valid_result = eva.do_valid(model)
                modelSelector.log(valid_result)
                indicator = "valid/auc"
                if not modelSelector.is_best_model(indicator) or valid_result["valid/auc"] < args.full_test_start:
                    wb_util.log(valid_result)
                else:
                    # best model!
                    test_result = eva.do_test(model)
                    wb_util.log({**valid_result, **test_result})
                    print(test_result)
                    model_util.delete_last_saved_model()
                    model_save_name = "auc[%.2f,%.2f]_ms[%.2f,%.2f,%.2f,%.2f]_map[%.2f,%.2f].pkl" % (
                        test_result["test/auc"],
                        test_result["test/auc_g"],
                        test_result["test/ms_v2f"],
                        test_result["test/ms_f2v"],
                        test_result["test/ms_v2f_g"],
                        test_result["test/ms_f2v_g"],
                        test_result["test/map_v2f"],
                        test_result["test/map_f2v"],
                    )
                    model_save_path = os.path.join(args.model_save_folder, args.project, args.name, model_save_name)
                    model_util.save_model(0, model, None, model_save_path)
                    json_path = model_save_path + ".json"
                    pickle_util.save_json(json_path, test_result)
                    print("save best model:", model_save_path)

                if modelSelector.should_stop(indicator, args.early_stop):
                    print("early_stop!")
                    print(model_util.history_array[-1])
                    return

                wb_util.init(args)


if __name__ == "__main__":
    parser = myparser.MyParser(epoch=100, batch_size=256, lr=5e-4, model_save_folder="outputs", early_stop=10, worker=4)
    parser.custom({
        "load_model": "",
        "batch_per_epoch": 5000,
        "eval_step": 100,
        "full_test_start": 80,

        # expert feature
        "face_features": "f_2plus1D_512,f_swin_512,f_mobile_512,f_dynamic_incept_512",
        "voice_features": "v_ecapa_192,v_resemble_256",

        # whether use position embedding：
        "use_pos_emb": False,

        # whether use modal token：
        "modal_token_std": -1.0,  # 0.05

        # transformer structure：
        "trans_input_dim": 512,
        "trans_n_head": 4,
        "trans_feedforward": 100,
        "trans_n_layer": 1,
        "trans_dropout": 0.5,
        "use_final_project_layer": False,

        # sbc:
        "big_batch": 512,
        "bc_mode": "sbc_4.0",  # sbc_Ratio、label、vanilla

        # loss:
        "infoNCE_temperature": 0.07,
    })
    parser.use_wb("project", "run")
    args = parser.parse()
    seed_util.set_seed(args.seed)

    # loader
    the_features = args.face_features.split(",") + args.voice_features.split(",")
    dataset = loader.DataSet(the_features, args.batch_per_epoch * args.batch_size,
                             args.big_batch,
                             args.batch_size, args.bc_mode)
    train_iter = DataLoader(dataset,
                            batch_size=None,
                            shuffle=True,
                            pin_memory=True,
                            worker_init_fn=worker_util.worker_init_fn,
                            num_workers=args.worker, )

    # model
    model = transformer.Model(the_features,
                              args.trans_input_dim,
                              args.trans_n_head,
                              args.trans_feedforward,
                              args.trans_n_layer,
                              args.trans_dropout,
                              args.modal_token_std,
                              args.use_final_project_layer,
                              args.use_pos_emb
                              )
    if len(args.load_model) > 0:
        print("load model", args.load_model)
        model_util.load_model(args.load_model, model, strict=False)

    model.cuda()

    # 3.loss
    loss_func = unsup_nce.InfoNCE(args.infoNCE_temperature)
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, lr=args.lr)

    # 4.eval
    eva = eval_util.EmbEva(the_features)
    modelSelector = model_selector.ModelSelector()
    train()
