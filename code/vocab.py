import jieba


if __name__ == "__main__":
    strs=["我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]
    seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
