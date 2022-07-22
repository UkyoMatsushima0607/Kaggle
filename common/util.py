"""共通処理
よく使い様な処理を関数化する
"""
import os

import pandas as pd

class CommonUtil:
  """共通クラス

  """

  def load_data(file_name):
    """ファイル読み込み

      dataフォルダからfile_nameに一致するファイルを読み込む

      Args:
        file_name: 拡張子含むファイル名

      Returns:
        DataFrame or TextParser or None

    """

    if not file_name:
      return None
    elif os.path.exists('hello.txt'):
      

