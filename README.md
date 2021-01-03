## 1. Environment

* WSL2(Ubuntu 20.04.1 LTS)
* Python 3.8.6
    * absl-py 0.11.0
    * astunparse 1.6.3
    * cachetools 4.1.1
    * certifi 2020.6.20
    * chardet 3.0.4
    * cloudpickle 1.6.0
    * cycler 0.10.0
    * decorator 4.4.2
    * fasttext 0.9.2
    * future 0.18.2
    * gast 0.3.3
    * gensim 3.8.3
    * google-auth 1.23.0
    * google-auth-oauthlib 0.4.2
    * google-pasta 0.2.0
    * grpcio 1.32.0
    * h5py 2.10.0
    * hyperopt 0.2.5
    * idna 2.10
    * joblib 0.17.0
    * Keras 2.4.3
    * Keras-Preprocessing 1.1.2
    * kiwisolver 1.3.0
    * Markdown 3.3.3
    * matplotlib 3.3.2
    * mecab-python3 1.0.2
    * neologdn 0.4
    * networkx 2.5
    * numpy 1.19.2
    * oauthlib 3.1.0
    * opt-einsum 3.3.0
    * pandas 1.1.3
    * Pillow 8.0.1
    * pip 20.2.4
    * protobuf 3.13.0
    * pyasn1 0.4.8
    * pyasn1-modules 0.2.8
    * pybind11 2.6.0
    * pyparsing 2.4.7
    * python-dateutil 2.8.1
    * pytz 2020.1
    * PyYAML 5.3.1
    * requests 2.24.0
    * requests-oauthlib 1.3.0
    * rsa 4.6
    * scikit-learn 0.23.2
    * scipy 1.5.3
    * setuptools 50.3.2
    * six 1.15.0
    * smart-open 3.0.0
    * tensorboard 2.4
    * tensorboard-plugin-wit 1.7.0
    * tensorflow 2.4.0
    * tensorflow-estimator 2.4.0rc0
    * termcolor 1.1.0
    * threadpoolctl 2.1.0
    * tqdm 4.51.0
    * unidic-lite 1.0.7
    * urllib3 1.25.11
    * Werkzeug 1.0.1
    * wheel 0.35.1
    * wrapt 1.12.1

## 2. Reference

* [『15Stepで踏破 自然言語処理アプリケーション開発入門 (StepUp!選書)』](https://bookmeter.com/books/14438482)
* [gensim](https://radimrehurek.com/gensim/auto_examples/index.html)
* [Keras](https://keras.io/guides/)
* [matplotlib](https://matplotlib.org/)
* [mecab](https://taku910.github.io/mecab/)
* [neologd](https://github.com/neologd/mecab-ipadic-neologd)
* [NumPy v1.19 Manual](https://numpy.org/doc/stable/)
* [pandas](https://pandas.pydata.org/docs/)
* [scikit-laern](https://scikit-learn.org/stable/user_guide.html)
* [SciPy](https://www.scipy.org/docs.html)
* [Regexp.ja](https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja)

## 3. Setup

### 3-1. Library

* If you have not install `lzma` in your environment, an error will occur in importing `pandas` as follows.

```bash
/home/username/.pyenv/versions/3.8.1/lib/python3.8/site-packages/pandas/compat/__init__.py:117: UserWarning: Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError.
  warnings.warn(msg)
  ```
To avoid it, install `lzma` in advance before you install Python.

For Debian / Ubuntu

```bash
$ sudo apt install liblzma-dev
```

* Install all required liblaries

```bash
$ pip install -r requirements.txt
```

* Install mecab to execute command line

For Debian / Ubuntu

```bash
$ sudo apt install mecab libmecab-dev mecab-ipadic-utf8
```

For MacOS

```bash
$ brew install mecab mecab-ipadic-utf8
```

* ipadic-NEologd is recommended, which is extended based on IPAdic and has been expandeding its vvocabulary crowling words appring on the internet.

Install
```bash
$ git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
$ cd mecab-ipadic-neologd
$ ./bin/install-mecab-ipadic-neologd -n -a
```

Checked the path where ipadic-NEologd is installed

```bash
$ echo `mecab-config --dicdir`"/mecab-ipadic-neologd"
```

Check the behaviours

```bash
$ mecab -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd
ラルクのメンバーはいつの間にかみんな五十路だったんだ
ラルク  名詞,固有名詞,人名,一般,*,*,ラルク,ラルク,ラルク
の      助詞,連体化,*,*,*,*,の,ノ,ノ
メンバー        名詞,一般,*,*,*,*,メンバー,メンバー,メンバー
は      助詞,係助詞,*,*,*,*,は,ハ,ワ
いつの間にか    副詞,一般,*,*,*,*,いつの間にか,イツノマニカ,イツノマニカ
みんな  名詞,代名詞,一般,*,*,*,みんな,ミンナ,ミンナ
五十路  名詞,一般,*,*,*,*,五十路,イソジ,イソジ
だっ    助動詞,*,*,*,特殊・ダ,連用タ接続,だ,ダッ,ダッ
た      助動詞,*,*,*,特殊・タ,基本形,た,タ,タ
ん      名詞,非自立,一般,*,*,*,ん,ン,ン
だ      助動詞,*,*,*,特殊・ダ,基本形,だ,ダ,ダ
EOS
```

### 3-2. Font

Install IPAPGothic for visibility of Japanese language in your terminal

Install

```bash
$ sudo apt install -y fonts-ipafont
```

Update font cache

```bash
$ fc-cache -fv
```

Check if the font is successfully installed

```bash
$ fc-list | grep -i ipa
```
