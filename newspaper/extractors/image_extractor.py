import logging
import urllib.parse
from copy import copy
import re
from typing import List, Optional, Tuple
import lxml
from PIL import Image, ImageFile
import requests
from newspaper import urls
import newspaper.parsers as parsers
from newspaper.network import session
from newspaper.configuration import Configuration
import newspaper.extractors.defines as defines
from newspaper.urls import urljoin_if_valid

log = logging.getLogger(__name__)


class ImageExtractor:
    """Extractor class for images in articles. Getting top image,
    image list, favicon, etc."""

    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.top_image: Optional[str] = None
        self.meta_image: Optional[str] = None
        self.images: List[str] = []
        self.favicon: Optional[str] = None
        self._chunksize = 1024

    def parse(
        self, doc: lxml.html.Element, top_node: lxml.html.Element, article_url: str
    ) -> None:
        """main method to extract images from a document

        Args:
            doc (lxml.html.Element): _description_
        """
        self.favicon = self._get_favicon(doc)

        self.meta_image = self._get_meta_image(doc)
        if self.meta_image:
            self.meta_image = urljoin_if_valid(article_url, self.meta_image)
        self.images = self._get_images(doc, top_node, article_url)  # Tried to use top_node, but images
        self.top_image = self._get_top_image(doc, top_node, article_url)

    def _get_favicon(self, doc: lxml.html.Element) -> str:
        """Extract the favicon from a website http://en.wikipedia.org/wiki/Favicon
        <link rel="shortcut icon" type="image/png" href="favicon.png" />
        <link rel="icon" type="image/png" href="favicon.png" />
        """
        meta = parsers.get_tags(
            doc, tag="link", attribs={"rel": "icon"}, attribs_match="substring"
        )
        if meta:
            favicon = parsers.get_attribute(meta[0], "href")
            return favicon or ""
        return ""

    def _get_meta_image(self, doc: lxml.html.Element) -> str:
        """Extract image from the meta tags of the document."""
        candidates: List[Tuple[str, int]] = []
        for elem in defines.META_IMAGE_TAGS:
            if "|" in elem["value"]:
                items = parsers.get_tags_regex(
                    doc, tag=elem["tag"], attribs={elem["attr"]: elem["value"]}
                )
            else:
                items = parsers.get_tags(
                    doc,
                    tag=elem["tag"],
                    attribs={elem["attr"]: elem["value"]},
                    attribs_match="exact",
                )

            candidates.extend((el.get(elem["content"]), elem["score"]) for el in items)

        candidates = [c for c in candidates if c[0]]

        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[0][0] if candidates else ""

    def _get_images(self, doc: lxml.html.Element, top_node: lxml.html.Element, article_url: str) -> List[str]:
        """
        Get all images in the document that meet the same criteria used for the top image.
        
        Args:
        - doc: lxml HTML document object.
        - top_node: The top image node to compare distances with.
        - article_url: The URL of the article, used for resolving relative URLs.
        
        Returns:
        - A list of valid image URLs.
        """
        
        def node_distance(node1, node2):
            """Calculate the distance between two nodes in the DOM."""
            path1 = node1.getroottree().getpath(node1).split("/")
            path2 = node2.getroottree().getpath(node2).split("/")
            for i, (step1, step2) in enumerate(zip(path1, path2)):
                if step1 != step2:
                    return len(path1[i:]) + len(path2[i:])
            return abs(len(path1) - len(path2))
        
        # List to hold valid image URLs
        valid_images = []
        
        # Iterate through all image tags in the document
        for img in parsers.get_tags(doc, tag="img"):
            # Skip if there's no src attribute or it's a data URI
            if not img.get("src") or img.get("src").startswith("data:"):
                continue
            
            # Check the size of the image before including it
            full_image_url = urljoin_if_valid(article_url, img.get("src"))
            if not self._check_image_size(full_image_url, article_url):
                continue
            
            # If top_node is provided, calculate the distance from it and prioritize closer images
            if top_node is not None:
                distance = node_distance(top_node, img)
                valid_images.append((full_image_url, distance))
            else:
                valid_images.append((full_image_url, 0))  # If no top_node, append with distance 0
        
        # Sort images by distance if top_node is used
        valid_images.sort(key=lambda x: x[1])
        
        # Return only the image URLs (filter out the distances)
        return [img[0] for img in valid_images]

    def _get_top_image(
        self, doc: lxml.html.Element, top_node: lxml.html.Element, article_url: str
    ) -> str:
        def node_distance(node1, node2):
            path1 = node1.getroottree().getpath(node1).split("/")
            path2 = node2.getroottree().getpath(node2).split("/")
            for i, (step1, step2) in enumerate(zip(path1, path2)):
                if step1 != step2:
                    return len(path1[i:]) + len(path2[i:])

            return abs(len(path1) - len(path2))

        if self.meta_image:
            if not self.config.fetch_images or self._check_image_size(
                self.meta_image, article_url
            ):
                return self.meta_image

        img_cand = []
        for img in parsers.get_tags(doc, tag="img"):
            if not img.get("src"):
                continue
            if img.get("src").startswith("data:"):
                continue

            if top_node is not None:
                distance = node_distance(top_node, img)
                img_cand.append((img, distance))
            else:
                if self._check_image_size(img.get("src"), article_url):
                    return img.get("src")

        img_cand.sort(key=lambda x: x[1])

        for img in img_cand:
            if self._check_image_size(img[0].get("src"), article_url):
                return img[0].get("src")

        return ""

    def _check_image_size(self, url: str, referer: Optional[str]) -> bool:
        img = self._fetch_image(
            url,
            referer,
        )
        if not img:
            return False

        width, height = img.size

        if self.config.top_image_settings["min_width"] > width:
            return False
        if self.config.top_image_settings["min_height"] > height:
            return False
        if self.config.top_image_settings["min_area"] > width * height:
            return False

        if (
            re.search(r"(logo|sprite)", url, re.IGNORECASE)
            and self.config.top_image_settings["min_area"] > width * height / 10
        ):
            return False

        return True

    def _fetch_image(self, url: str, referer: Optional[str]) -> Optional[Image.Image]:
        def clean_url(url):
            """Url quotes unicode data out of urls"""
            if not isinstance(url, str):
                return url

            url = url.encode("utf8")
            url = "".join(
                [
                    urllib.parse.quote(c) if ord(c) >= 127 else c
                    for c in url.decode("utf-8")
                ]
            )
            return url

        requests_params = copy(self.config.requests_params)
        requests_params["headers"]["Referer"] = referer
        max_retries = self.config.top_image_settings["max_retries"]

        cur_try = 0
        url = clean_url(url)
        if not url or not url.startswith(("http://", "https://")):
            return None

        response = None
        while True:
            try:
                response = session.get(
                    url,
                    stream=True,
                    **requests_params,
                )

                content_type = response.headers.get("Content-Type")

                if not content_type or "image" not in content_type.lower():
                    return None

                p = ImageFile.Parser()
                new_data = response.raw.read(self._chunksize)
                while not p.image and new_data:
                    try:
                        p.feed(new_data)
                    except (IOError, ValueError) as e:
                        log.warning(
                            "error %s while fetching: %s refer: %s",
                            str(e),
                            url,
                            requests_params["headers"].get("Referer"),
                        )
                        return None
                    except Exception as e:
                        # For some favicon.ico images, the image is so small
                        # that our PIL feed() method fails a length test.
                        is_favicon = urls.url_to_filetype(url) == "ico"
                        if not is_favicon:
                            raise e
                        return None
                    new_data = response.raw.read(self._chunksize)
                return p.image
            except requests.exceptions.RequestException:
                cur_try += 1
                if cur_try >= max_retries:
                    log.warning(
                        "error while fetching: %s refer: %s",
                        url,
                        requests_params["headers"].get("Referer"),
                    )
                    return None
            finally:
                if response is not None:
                    response.raw.close()
                    if response.raw._connection:
                        response.raw._connection.close()
