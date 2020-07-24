## Environment

* WSL(Ubuntu 18.04.4 LTS)
* Python 3.8.5
  * gensim 3.8.3
  * Keras 2.4.3
  * matplotlib 3.3.0
  * mecab-python3 1.0.1
  * neologdn 0.4
  * numpy 1.19.1
  * pandas 1.0.5
  * scikit-learn 0.23.1
  * scipy 1.4.1
  * tensorflow 2.2.0
  * unidic-lite 1.0.6

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
sudo apt install liblzma-dev
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
$ fc-cache -fv
```

Update font cache

```bash
$ fc-cache -fv
```

Check if the font is successfully installed

```bash
$ fc-list | grep -i ipa
```
