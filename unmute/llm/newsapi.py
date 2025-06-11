import logging
import os

import requests
from pydantic import BaseModel

from unmute.cache import CacheError, get_cache

logger = logging.getLogger(__name__)

newsapi_api_key = os.environ.get("NEWSAPI_API_KEY")


class Source(BaseModel):
    id: str | None
    name: str


class Article(BaseModel):
    source: Source
    author: str | None
    title: str
    description: str | None
    # Omit the URLs because we don't need them, save space
    # url: HttpUrl
    # urlToImage: HttpUrl | None
    publishedAt: str
    content: str | None


class NewsResponse(BaseModel):
    status: str
    totalResults: int
    articles: list[Article]


if not newsapi_api_key:
    logger.warning(
        "NEWSAPI_API_KEY is not set. News API functionality will be disabled."
    )


cache = get_cache("newsapi", ttl_seconds=60 * 60 * 4)  # 4 hours
CACHE_KEY = "news"


def get_news_without_caching() -> NewsResponse | None:
    if not newsapi_api_key:
        return None

    logger.info("Fetching news from News API")
    response = requests.get(
        "https://newsapi.org/v2/everything?sources=the-verge",
        headers={"Authorization": newsapi_api_key},
    )
    response.raise_for_status()
    news_response = NewsResponse(**response.json())

    return news_response


def get_news() -> NewsResponse | None:
    try:
        cached_news_raw = cache.get(CACHE_KEY)
    except CacheError as e:
        logger.error(f"Failed to fetch news from cache: {e}")
        # Refuse to query because that would mean we have to query the API every time
        return None

    cached_news = (
        NewsResponse.model_validate_json(cached_news_raw) if cached_news_raw else None
    )

    if cached_news is None:
        try:
            cached_news = get_news_without_caching()
            if cached_news:
                cache.set(CACHE_KEY, cached_news.model_dump_json())

        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
            return None

    return cached_news


if __name__ == "__main__":
    news = get_news()
    if news:
        print(news.model_dump_json(indent=2))
