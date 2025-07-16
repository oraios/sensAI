import io
import logging
import os
from typing import Sequence, Optional, Tuple, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    import pandas as pd

log = logging.getLogger(__name__)


class ResultWriter:
    log = log.getChild(__qualname__)

    def __init__(self, result_dir, filename_prefix="", enabled: bool = True, close_figures: bool = False):
        """
        :param result_dir:
        :param filename_prefix:
        :param enabled: whether the result writer is enabled; if it is not, it will create neither files nor directories
        :param close_figures: whether to close figures that are passed by default
        """
        self.result_dir = result_dir
        self.filename_prefix = filename_prefix
        self.enabled = enabled
        self.close_figures_default = close_figures
        if self.enabled:
            os.makedirs(result_dir, exist_ok=True)

    def child_with_added_prefix(self, prefix: str) -> "ResultWriter":
        """
        Creates a derived result writer with an added prefix, i.e. the given prefix is appended to this
        result writer's prefix

        :param prefix: the prefix to append
        :return: a new writer instance
        """
        return ResultWriter(self.result_dir, filename_prefix=self.filename_prefix + prefix, enabled=self.enabled,
            close_figures=self.close_figures_default)

    def child_for_subdirectory(self, dir_name: str) -> "ResultWriter":
        result_dir = os.path.join(self.result_dir, dir_name)
        return ResultWriter(result_dir, filename_prefix=self.filename_prefix, enabled=self.enabled,
            close_figures=self.close_figures_default)

    def path(self, filename_suffix: str, extension_to_add=None, valid_other_extensions: Optional[Sequence[str]] = None) -> str:
        """
        :param filename_suffix: the suffix to add (which may or may not already include a file extension)
        :param extension_to_add: if not None, the file extension to add (without the leading ".") unless
            the extension to add or one of the extenions in valid_extensions is already present
        :param valid_other_extensions: a sequence of valid other extensions (without the "."), only
            relevant if extensionToAdd is specified
        :return: the full path
        """
        # replace forbidden characters
        filename_suffix = filename_suffix.replace(">=", "gte").replace(">", "gt")

        if extension_to_add is not None:
            add_ext = True
            valid_extensions = set(valid_other_extensions) if valid_other_extensions is not None else set()
            valid_extensions.add(extension_to_add)
            if valid_extensions is not None:
                for ext in valid_extensions:
                    if filename_suffix.endswith("." + ext):
                        add_ext = False
                        break
            if add_ext:
                filename_suffix += "." + extension_to_add
        path = os.path.join(self.result_dir, f"{self.filename_prefix}{filename_suffix}")
        return path

    def write_text_file(self, filename_suffix: str, content: str):
        p = self.path(filename_suffix, extension_to_add="txt")
        if self.enabled:
            self.log.info(f"Saving text file {p}")
            with open(p, "w") as f:
                f.write(content)
        return p

    def write_text_file_lines(self, filename_suffix: str, lines: List[str]):
        p = self.path(filename_suffix, extension_to_add="txt")
        if self.enabled:
            self.log.info(f"Saving text file {p}")
            write_text_file_lines(lines, p)
        return p

    def write_data_frame_text_file(self, filename_suffix: str, df: "pd.DataFrame"):
        p = self.path(filename_suffix, extension_to_add="df.txt", valid_other_extensions="txt")
        if self.enabled:
            self.log.info(f"Saving data frame text file {p}")
            with open(p, "w") as f:
                f.write(df.to_string())
        return p

    def write_data_frame_csv_file(self, filename_suffix: str, df: "pd.DataFrame", index=True, header=True):
        p = self.path(filename_suffix, extension_to_add="csv")
        if self.enabled:
            self.log.info(f"Saving data frame CSV file {p}")
            df.to_csv(p, index=index, header=header)
        return p

    def write_data_frame_excel_file(self, filename_suffix: str, df: "pd.DataFrame", index=True, header=True, sheet_name="Sheet1"):
        """
        Saves the given data frame to an Excel spreadsheet.

        Requires that an appropriate engine be installed, e.g. openpyxl.

        :param filename_suffix: the filename suffix (or full basename for the case where no prefix was specified at construction),
            to which the extension .xlsx will be added by default if no extension is given
        :param df: the data frame to save
        :param index: whether to save the index
        :param header: whether to save the header row
        :param sheet_name: the name of the spreadsheet
        :return: the path to the saved file
        """
        p = self.path(filename_suffix, extension_to_add="xlsx")
        if self.enabled:
            self.log.info(f"Saving data frame Excel file {p}")
            df.to_excel(p, index=index, header=header, sheet_name=sheet_name)
        return p

    def write_figure(self, filename_suffix: str, fig: "plt.Figure", close_figure: Optional[bool] = None):
        """
        :param filename_suffix: the filename suffix, which may or may not include a file extension, valid extensions being {"png", "jpg"}
        :param fig: the figure to save
        :param close_figure: whether to close the figure after having saved it; if None, use default passed at construction
        :return: the path to the file that was written (or would have been written if the writer was enabled)
        """
        from matplotlib import pyplot as plt
        p = self.path(filename_suffix, extension_to_add="png", valid_other_extensions=("jpg",))
        if self.enabled:
            self.log.info(f"Saving figure {p}")
            fig.savefig(p, bbox_inches="tight")
            must_close_figure = close_figure if close_figure is not None else self.close_figures_default
            if must_close_figure:
                plt.close(fig)
        return p

    def write_figures(self, figures: Sequence[Tuple[str, "plt.Figure"]], close_figures=False):
        for name, fig in figures:
            self.write_figure(name, fig, close_figure=close_figures)

    def write_pickle(self, filename_suffix: str, obj: Any):
        from .pickle import dump_pickle
        p = self.path(filename_suffix, extension_to_add="pickle")
        if self.enabled:
            self.log.info(f"Saving pickle {p}")
            dump_pickle(obj, p)
        return p


def write_text_file_lines(lines: List[str], path):
    """
    :param lines: the lines to write (without a trailing newline, which will be added)
    :param path: the path of the text file to write to
    """
    with open(path, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def read_text_file_lines(path, strip=True, skip_empty=True) -> List[str]:
    """
    :param path: the path of the text file to read from
    :param strip: whether to strip each line, removing whitespace/newline characters
    :param skip_empty: whether to skip any lines that are empty (after stripping)
    :return: the list of lines
    """
    lines = []
    with open(path, "r") as f:
        for line in f.readlines():
            if strip:
                line = line.strip()
            if not skip_empty or line != "":
                lines.append(line)
    return lines


def is_s3_path(path: str):
    return path.startswith("s3://")


class S3Object:
    def __init__(self, path):
        assert is_s3_path(path)
        self.path = path
        self.bucket, self.object = self.path[5:].split("/", 1)

    class OutputFile:
        def __init__(self, s3_object: "S3Object"):
            self.s3Object = s3_object
            self.buffer = io.BytesIO()

        def write(self, obj: bytes):
            self.buffer.write(obj)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.s3Object.put(self.buffer.getvalue())

    def get_file_content(self):
        return self._get_s3_object().get()['Body'].read()

    def open_file(self, mode):
        assert mode in ("wb", "rb")
        if mode == "rb":
            content = self.get_file_content()
            return io.BytesIO(content)

        elif mode == "wb":
            return self.OutputFile(self)

        else:
            raise ValueError(mode)

    def put(self, obj: bytes):
        self._get_s3_object().put(Body=obj)

    def _get_s3_object(self):
        import boto3
        session = boto3.session.Session(profile_name=os.getenv("AWS_PROFILE"))
        s3 = session.resource("s3")
        return s3.Bucket(self.bucket).Object(self.object)


def create_path(root: str, *path_elems: str, is_dir: bool, make_dirs: bool = False) -> str:
    path = os.path.join(root, *path_elems)
    if make_dirs:
        dir_path = path if is_dir else os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
    return path


def create_file_path(root, *path_elems, make_dirs: bool = False) -> str:
    return create_path(root, *path_elems, is_dir=False, make_dirs=make_dirs)


def create_dir_path(root, *path_elems, make_dirs: bool = False) -> str:
    return create_path(root, *path_elems, is_dir=True, make_dirs=make_dirs)


def filename_compatible(fn: str, replacement: str = "-") -> str:
    """
    Converts the given string into a string that can be used in a filename

    :param fn: original filename, which may or may not be compatible with common filenames
    :param replacement: default character to use as a replacement
    :return: adapted filename
    """
    # Replace prohibited characters
    # Windows prohibits: \ / : * ? " < > |
    # Unix/Linux prohibits: /
    # macOS prohibits: / and :
    replacements = {
        ">": "gt",
        "<": "lt",
        "/": replacement,
        "\\": replacement,
        ":": replacement,
        "*": replacement,
        "?": replacement,
        "\"": replacement,
        "|": replacement
    }

    result = fn
    for char, repl in replacements.items():
        result = result.replace(char, repl)

    return result
